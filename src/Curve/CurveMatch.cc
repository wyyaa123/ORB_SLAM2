#include "Curve/CurveMatcher.h"

#include "Curve/BezierCurve.h"
#include "Curve/Hungarian.h"
#include "Curve/MapCurve.h"
#include "Frame.h"
#include "KeyFrame.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ORB_SLAM2
{
    const double kInvalidCost = 1e6;
    const size_t kMaxSimilaritySamples = 96;
    const float kProjectedSampleSpacing = 3.0f;
    const float kMaxMeanDirectionCost = 0.40f;
    const float kFragmentEndpointDistance = 12.0f;
    const float kFragmentTangentAlignment = 0.75f;

    static std::string DiagnosticNumber(const double value, const int precision = 2)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        return stream.str();
    }

    CurveSampleRef::CurveSampleRef(const size_t curveIndexValue, const size_t sampleIndexValue)
        : curveIndex(curveIndexValue), sampleIndex(sampleIndexValue)
    {
    }

    ProjectedMapCurve::ProjectedMapCurve() : pMapCurve(NULL)
    {
    }

    CandidateStatistics::CandidateStatistics() : matchedMapSamples(0)
    {
    }

    void CandidateStatistics::AddHit(const size_t observedSampleIndex)
    {
        ++matchedMapSamples;
        matchedObservedSamples.insert(observedSampleIndex);
    }

    CurveSimilarity::CurveSimilarity() : cost(kInvalidCost), meanDistance(kInvalidCost), meanDirectionCost(1.0), dtwCost(kInvalidCost), minMapIndex(-1), maxMapIndex(-1), valid(false)
    {
    }

    AcceptedFragment::AcceptedFragment(const size_t curveIndexValue, const CurveSimilarity &similarityValue)
        : curveIndex(curveIndexValue), similarity(similarityValue)
    {
    }

    CurveSampleGrid::CurveSampleGrid(const float minX, const float minY, const float maxX, const float maxY, const float cellSize)
        : mMinX(minX), mMinY(minY), mCellSize(std::max(1.0f, cellSize)), mColumns(std::max(1, static_cast<int>(std::ceil((maxX - minX) / mCellSize)))), mRows(std::max(1, static_cast<int>(std::ceil((maxY - minY) / mCellSize)))), mCells(static_cast<size_t>(mColumns * mRows))
    {
    }

    void CurveSampleGrid::Insert(const cv::Point2f &point, const CurveSampleRef &sample)
    {
        const int column = ColumnFor(point.x);
        const int row = RowFor(point.y);
        if (!IsValidCell(column, row))
            return;
        mCells[CellIndex(column, row)].push_back(sample);
    }

    void CurveSampleGrid::Query(const cv::Point2f &center, const float radius, std::vector<CurveSampleRef> &samples) const
    {
        samples.clear();

        const int firstColumn = std::max(0, ColumnFor(center.x - radius));
        const int lastColumn = std::min(mColumns - 1, ColumnFor(center.x + radius));
        const int firstRow = std::max(0, RowFor(center.y - radius));
        const int lastRow = std::min(mRows - 1, RowFor(center.y + radius));

        if (firstColumn > lastColumn || firstRow > lastRow)
            return;

        for (int row = firstRow; row <= lastRow; ++row)
        {
            for (int column = firstColumn; column <= lastColumn; ++column)
            {
                const std::vector<CurveSampleRef> &cell = mCells[CellIndex(column, row)];
                samples.insert(samples.end(), cell.begin(), cell.end());
            }
        }
    }

    int CurveSampleGrid::ColumnFor(const float x) const
    {
        return static_cast<int>(std::floor((x - mMinX) / mCellSize));
    }

    int CurveSampleGrid::RowFor(const float y) const
    {
        return static_cast<int>(std::floor((y - mMinY) / mCellSize));
    }

    bool CurveSampleGrid::IsValidCell(const int column, const int row) const
    {
        return column >= 0 && column < mColumns && row >= 0 && row < mRows;
    }

    size_t CurveSampleGrid::CellIndex(const int column, const int row) const
    {
        return static_cast<size_t>(row * mColumns + column);
    }

    DirectedCurveMetrics::DirectedCurveMetrics() : meanDistance(kInvalidCost), meanDirectionCost(1.0), minReferenceIndex(-1), maxReferenceIndex(-1)
    {
    }

    cv::Point2f CurveMatcher::ToPoint2f(const orderedEdgePoint &point)
    {
        return cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y));
    }

    bool CurveMatcher::ProjectWorldPoint(const cv::Point3d &worldPoint, const Frame &frame, cv::Point2f &imagePoint)
    {
        const cv::Mat &Tcw = frame.mTcw;
        const float worldX = static_cast<float>(worldPoint.x);
        const float worldY = static_cast<float>(worldPoint.y);
        const float worldZ = static_cast<float>(worldPoint.z);

        const float cameraX = Tcw.at<float>(0, 0) * worldX + Tcw.at<float>(0, 1) * worldY + Tcw.at<float>(0, 2) * worldZ + Tcw.at<float>(0, 3);
        const float cameraY = Tcw.at<float>(1, 0) * worldX + Tcw.at<float>(1, 1) * worldY + Tcw.at<float>(1, 2) * worldZ + Tcw.at<float>(1, 3);
        const float cameraZ = Tcw.at<float>(2, 0) * worldX + Tcw.at<float>(2, 1) * worldY + Tcw.at<float>(2, 2) * worldZ + Tcw.at<float>(2, 3);

        if (!std::isfinite(cameraX) || !std::isfinite(cameraY) || !std::isfinite(cameraZ) || cameraZ <= 0.0f)
        {
            return false;
        }

        const float inverseZ = 1.0f / cameraZ;
        imagePoint.x = frame.fx * cameraX * inverseZ + frame.cx;
        imagePoint.y = frame.fy * cameraY * inverseZ + frame.cy;
        return std::isfinite(imagePoint.x) && std::isfinite(imagePoint.y);
    }

    bool CurveMatcher::ProjectWorldPoint(const cv::Point3d &worldPoint, KeyFrame *pKF, cv::Point2f &imagePoint)
    {
        if (!pKF)
            return false;
        const cv::Mat Tcw = pKF->GetPose();
        if (Tcw.empty())
            return false;
        const float worldX = static_cast<float>(worldPoint.x);
        const float worldY = static_cast<float>(worldPoint.y);
        const float worldZ = static_cast<float>(worldPoint.z);
        const float cameraX = Tcw.at<float>(0, 0) * worldX + Tcw.at<float>(0, 1) * worldY + Tcw.at<float>(0, 2) * worldZ + Tcw.at<float>(0, 3);
        const float cameraY = Tcw.at<float>(1, 0) * worldX + Tcw.at<float>(1, 1) * worldY + Tcw.at<float>(1, 2) * worldZ + Tcw.at<float>(1, 3);
        const float cameraZ = Tcw.at<float>(2, 0) * worldX + Tcw.at<float>(2, 1) * worldY + Tcw.at<float>(2, 2) * worldZ + Tcw.at<float>(2, 3);
        if (!std::isfinite(cameraX) || !std::isfinite(cameraY) || !std::isfinite(cameraZ) || cameraZ <= 0.0f)
            return false;
        const float inverseZ = 1.0f / cameraZ;
        imagePoint.x = pKF->fx * cameraX * inverseZ + pKF->cx;
        imagePoint.y = pKF->fy * cameraY * inverseZ + pKF->cy;
        return std::isfinite(imagePoint.x) && std::isfinite(imagePoint.y);
    }

    void CurveMatcher::AppendResampledSegment(const cv::Point2f &first, const cv::Point2f &second, std::vector<cv::Point2f> &samples)
    {
        const float length = cv::norm(second - first);
        if (length <= 1e-3f)
            return;

        const int intervalCount = std::max(1, static_cast<int>(std::ceil(length / kProjectedSampleSpacing)));

        for (int interval = 1; interval <= intervalCount; ++interval)
        {
            const float ratio = static_cast<float>(interval) / intervalCount;
            samples.push_back(first + ratio * (second - first));
        }
    }

    bool CurveMatcher::ProjectMapCurve(MapCurve *pMapCurve, const Frame &frame, const float margin, ProjectedMapCurve &projected)
    {
        projected = ProjectedMapCurve();
        if (!pMapCurve || pMapCurve->isBad() || frame.mTcw.empty())
            return false;

        projected.pMapCurve = pMapCurve;
        const std::vector<cv::Point3d> worldPoints = pMapCurve->GetCurvePoints();

        bool hasPreviousPoint = false;
        cv::Point2f previousPoint;
        for (const cv::Point3d &worldPoint : worldPoints)
        {
            cv::Point2f currentPoint;
            if (!ProjectWorldPoint(worldPoint, frame, currentPoint) || currentPoint.x < frame.mnMinX - margin || currentPoint.x > frame.mnMaxX + margin || currentPoint.y < frame.mnMinY - margin || currentPoint.y > frame.mnMaxY + margin)
            {
                hasPreviousPoint = false;
                continue;
            }

            if (!hasPreviousPoint)
                projected.samples.push_back(currentPoint);
            else
                AppendResampledSegment(previousPoint, currentPoint, projected.samples);

            previousPoint = currentPoint;
            hasPreviousPoint = true;
        }

        return projected.samples.size() >= 2;
    }

    bool CurveMatcher::ProjectMapCurve(MapCurve *pMapCurve, KeyFrame *pKF, const float margin, ProjectedMapCurve &projected)
    {
        projected = ProjectedMapCurve();
        if (!pMapCurve || !pKF || pMapCurve->isBad() || pKF->isBad())
            return false;
        projected.pMapCurve = pMapCurve;
        const std::vector<cv::Point3d> worldPoints = pMapCurve->GetCurvePoints();
        bool hasPreviousPoint = false;
        cv::Point2f previousPoint;
        for (const cv::Point3d &worldPoint : worldPoints)
        {
            cv::Point2f currentPoint;
            if (!ProjectWorldPoint(worldPoint, pKF, currentPoint) || currentPoint.x < pKF->mnMinX - margin || currentPoint.x > pKF->mnMaxX + margin || currentPoint.y < pKF->mnMinY - margin || currentPoint.y > pKF->mnMaxY + margin)
            {
                hasPreviousPoint = false;
                continue;
            }
            if (!hasPreviousPoint)
                projected.samples.push_back(currentPoint);
            else
                AppendResampledSegment(previousPoint, currentPoint, projected.samples);
            previousPoint = currentPoint;
            hasPreviousPoint = true;
        }
        return projected.samples.size() >= 2;
    }

    std::vector<cv::Point2f> CurveMatcher::LimitSamples(const std::vector<cv::Point2f> &points, const size_t maximumSamples)
    {
        if (points.size() <= maximumSamples)
            return points;

        std::vector<cv::Point2f> samples;
        samples.reserve(maximumSamples);
        for (size_t index = 0; index < maximumSamples; ++index)
        {
            const size_t sourceIndex = static_cast<size_t>(std::round(static_cast<double>(index) * (points.size() - 1) / (maximumSamples - 1)));
            samples.push_back(points[sourceIndex]);
        }
        return samples;
    }

    cv::Point2f CurveMatcher::CurveTangentAt(const std::vector<cv::Point2f> &points, const size_t index)
    {
        if (points.size() < 2)
            return cv::Point2f();

        const size_t first = index == 0 ? 0 : index - 1;
        const size_t second = std::min(index + 1, points.size() - 1);
        cv::Point2f tangent = points[second] - points[first];
        const float length = cv::norm(tangent);
        if (length <= 1e-6f)
            return cv::Point2f();
        return tangent * (1.0f / length);
    }

    double CurveMatcher::LocalAlignmentCost(const std::vector<cv::Point2f> &reference, const size_t referenceIndex, const std::vector<cv::Point2f> &query, const size_t queryIndex)
    {
        const double distance = cv::norm(reference[referenceIndex] - query[queryIndex]);
        const cv::Point2f referenceTangent = CurveTangentAt(reference, referenceIndex);
        const cv::Point2f queryTangent = CurveTangentAt(query, queryIndex);

        double directionCost = 1.0;
        const float referenceLength = cv::norm(referenceTangent);
        const float queryLength = cv::norm(queryTangent);
        if (referenceLength > 0.0f && queryLength > 0.0f)
        {
            directionCost = 1.0 - std::abs(static_cast<double>(referenceTangent.dot(queryTangent)));
        }

        return distance + 4.0 * directionCost;
    }

    double CurveMatcher::SubsequenceDtwCost(const std::vector<cv::Point2f> &reference, const std::vector<cv::Point2f> &query)
    {
        if (reference.size() < 2 || query.size() < 2)
            return kInvalidCost;

        std::vector<double> previous(reference.size());
        std::vector<double> current(reference.size(), kInvalidCost);
        for (size_t referenceIndex = 0; referenceIndex < reference.size(); ++referenceIndex)
        {
            previous[referenceIndex] = LocalAlignmentCost(reference, referenceIndex, query, 0);
        }

        const size_t maximumReferenceAdvance = 4;
        for (size_t queryIndex = 1; queryIndex < query.size(); ++queryIndex)
        {
            std::fill(current.begin(), current.end(), kInvalidCost);
            for (size_t referenceIndex = 0; referenceIndex < reference.size(); ++referenceIndex)
            {
                double previousBest = previous[referenceIndex];
                const size_t firstPrevious = referenceIndex > maximumReferenceAdvance ? referenceIndex - maximumReferenceAdvance : 0;
                for (size_t previousIndex = firstPrevious; previousIndex < referenceIndex; ++previousIndex)
                {
                    previousBest = std::min(previousBest, previous[previousIndex]);
                }

                current[referenceIndex] = previousBest + LocalAlignmentCost(reference, referenceIndex, query, queryIndex);
            }
            previous.swap(current);
        }

        return *std::min_element(previous.begin(), previous.end()) / query.size();
    }

    double CurveMatcher::OrientationIndependentDtw(const std::vector<cv::Point2f> &reference, const std::vector<cv::Point2f> &query)
    {
        const double forwardCost = SubsequenceDtwCost(reference, query);
        const std::vector<cv::Point2f> reversedQuery(query.rbegin(), query.rend());
        const double reverseCost = SubsequenceDtwCost(reference, reversedQuery);
        return std::min(forwardCost, reverseCost);
    }

    DirectedCurveMetrics CurveMatcher::ComputeDirectedMetrics(const std::vector<cv::Point2f> &reference, const std::vector<cv::Point2f> &query)
    {
        DirectedCurveMetrics metrics;
        if (reference.size() < 2 || query.size() < 2)
            return metrics;

        double distanceSum = 0.0;
        double directionCostSum = 0.0;
        size_t directionCount = 0;
        int minimumReferenceIndex = static_cast<int>(reference.size());
        int maximumReferenceIndex = -1;

        for (size_t queryIndex = 0; queryIndex < query.size(); ++queryIndex)
        {
            size_t nearestReferenceIndex = 0;
            float nearestDistanceSquared = std::numeric_limits<float>::max();
            for (size_t referenceIndex = 0; referenceIndex < reference.size(); ++referenceIndex)
            {
                const cv::Point2f delta = query[queryIndex] - reference[referenceIndex];
                const float distanceSquared = delta.dot(delta);
                if (distanceSquared < nearestDistanceSquared)
                {
                    nearestDistanceSquared = distanceSquared;
                    nearestReferenceIndex = referenceIndex;
                }
            }

            distanceSum += std::sqrt(nearestDistanceSquared);
            minimumReferenceIndex = std::min(minimumReferenceIndex, static_cast<int>(nearestReferenceIndex));
            maximumReferenceIndex = std::max(maximumReferenceIndex, static_cast<int>(nearestReferenceIndex));

            const cv::Point2f referenceTangent = CurveTangentAt(reference, nearestReferenceIndex);
            const cv::Point2f queryTangent = CurveTangentAt(query, queryIndex);
            if (cv::norm(referenceTangent) > 0.0f && cv::norm(queryTangent) > 0.0f)
            {
                directionCostSum += 1.0 - std::abs(static_cast<double>(referenceTangent.dot(queryTangent)));
                ++directionCount;
            }
        }

        metrics.meanDistance = distanceSum / query.size();
        metrics.meanDirectionCost = directionCount > 0 ? directionCostSum / directionCount : 1.0;
        metrics.minReferenceIndex = minimumReferenceIndex;
        metrics.maxReferenceIndex = maximumReferenceIndex;
        return metrics;
    }

    CurveSimilarity CurveMatcher::ComputeCurveSimilarity(const ProjectedMapCurve &mapCurve, const std::vector<cv::Point2f> &observedCurve, const float maximumMeanDistance)
    {
        CurveSimilarity result;
        const std::vector<cv::Point2f> mapSamples = LimitSamples(mapCurve.samples, kMaxSimilaritySamples);
        const std::vector<cv::Point2f> observedSamples = LimitSamples(observedCurve, kMaxSimilaritySamples);
        if (mapSamples.size() < 2 || observedSamples.size() < 2)
            return result;

        const DirectedCurveMetrics observedToMap = ComputeDirectedMetrics(mapSamples, observedSamples);
        const DirectedCurveMetrics mapToObserved = ComputeDirectedMetrics(observedSamples, mapSamples);

        result.meanDistance = std::min(observedToMap.meanDistance, mapToObserved.meanDistance);
        result.meanDirectionCost = std::min(observedToMap.meanDirectionCost, mapToObserved.meanDirectionCost);
        if (result.meanDistance > maximumMeanDistance || result.meanDirectionCost > kMaxMeanDirectionCost)
        {
            return result;
        }

        result.dtwCost = std::min(OrientationIndependentDtw(mapSamples, observedSamples), OrientationIndependentDtw(observedSamples, mapSamples));

        result.cost = 0.55 * result.dtwCost + 0.30 * result.meanDistance + 0.15 * result.meanDirectionCost;
        result.minMapIndex = observedToMap.minReferenceIndex;
        result.maxMapIndex = observedToMap.maxReferenceIndex;
        result.valid = std::isfinite(result.cost);
        return result;
    }

    bool CurveMatcher::EndpointsAreContinuous(const std::vector<cv::Point2f> &first, const std::vector<cv::Point2f> &second)
    {
        if (first.size() < 2 || second.size() < 2)
            return false;

        const size_t firstEndpoints[2] = {0, first.size() - 1};
        const size_t secondEndpoints[2] = {0, second.size() - 1};
        float bestDistanceSquared = std::numeric_limits<float>::max();
        size_t bestFirst = 0;
        size_t bestSecond = 0;

        for (const size_t firstIndex : firstEndpoints)
        {
            for (const size_t secondIndex : secondEndpoints)
            {
                const cv::Point2f delta = first[firstIndex] - second[secondIndex];
                const float distanceSquared = delta.dot(delta);
                if (distanceSquared < bestDistanceSquared)
                {
                    bestDistanceSquared = distanceSquared;
                    bestFirst = firstIndex;
                    bestSecond = secondIndex;
                }
            }
        }

        if (bestDistanceSquared > kFragmentEndpointDistance * kFragmentEndpointDistance)
        {
            return false;
        }

        const cv::Point2f firstTangent = CurveTangentAt(first, bestFirst);
        const cv::Point2f secondTangent = CurveTangentAt(second, bestSecond);
        return cv::norm(firstTangent) > 0.0f && cv::norm(secondTangent) > 0.0f && std::abs(firstTangent.dot(secondTangent)) >= kFragmentTangentAlignment;
    }

    bool CurveMatcher::FragmentIsCompatible(const size_t candidateCurveIndex, const CurveSimilarity &candidateSimilarity, const std::vector<AcceptedFragment> &acceptedFragments, const Frame &frame, const std::vector<std::vector<cv::Point2f>> &observedCurves, const size_t projectedMapSampleCount)
    {
        const BezierCurve &candidateCurve = frame.mvBezierCurves[candidateCurveIndex];
        const int candidateSpan = std::max(1, candidateSimilarity.maxMapIndex - candidateSimilarity.minMapIndex + 1);
        const int maximumGap = std::max(4, static_cast<int>(0.25f * projectedMapSampleCount));

        // A depth discontinuity on the same source edge is a hard conflict.
        // Check it before ordinary endpoint compatibility so a fragment from a
        // different edge cannot indirectly bridge the two depth groups.
        for (const AcceptedFragment &accepted : acceptedFragments)
        {
            const BezierCurve &acceptedCurve =
                frame.mvBezierCurves[accepted.curveIndex];
            if (candidateCurve.isDepthDisconnectedFrom(acceptedCurve))
                return false;
        }

        for (const AcceptedFragment &accepted : acceptedFragments)
        {
            const CurveSimilarity &acceptedSimilarity = accepted.similarity;
            const int acceptedSpan = std::max(1, acceptedSimilarity.maxMapIndex - acceptedSimilarity.minMapIndex + 1);
            const int overlapBegin = std::max(candidateSimilarity.minMapIndex, acceptedSimilarity.minMapIndex);
            const int overlapEnd = std::min(candidateSimilarity.maxMapIndex, acceptedSimilarity.maxMapIndex);
            const int overlap = std::max(0, overlapEnd - overlapBegin + 1);
            const int allowedOverlap = std::max(1, static_cast<int>(std::ceil(0.15f * std::min(candidateSpan, acceptedSpan))));
            if (overlap > allowedOverlap)
                continue;

            int gap = 0;
            if (candidateSimilarity.maxMapIndex < acceptedSimilarity.minMapIndex)
            {
                gap = acceptedSimilarity.minMapIndex - candidateSimilarity.maxMapIndex;
            }
            else if (acceptedSimilarity.maxMapIndex < candidateSimilarity.minMapIndex)
            {
                gap = candidateSimilarity.minMapIndex - acceptedSimilarity.maxMapIndex;
            }
            if (gap > maximumGap)
                continue;

            const BezierCurve &acceptedCurve = frame.mvBezierCurves[accepted.curveIndex];
            const bool sameSourceEdge = candidateCurve.hasEdgeChain() && acceptedCurve.hasEdgeChain() && candidateCurve.edgeChainId == acceptedCurve.edgeChainId;

            if (sameSourceEdge || EndpointsAreContinuous(observedCurves[candidateCurveIndex], observedCurves[accepted.curveIndex]))
            {
                return true;
            }
        }

        return false;
    }

    CurveMatcher::CurveMatcher(const CurveConfigPtr &curveConfig) : mCurveConfig(curveConfig)
    {
    }

    void CurveMatcher::BuildFixedSampleCorrespondences(Frame &frame)
    {
        frame.mvCurveSampleCorrespondences.clear();
        if (!mCurveConfig || frame.mTcw.empty() || frame.mvBezierCurves.empty())
            return;

        std::unordered_map<MapCurve *, std::vector<size_t>> observedCurvesByMapCurve;
        for (size_t curveIndex = 0; curveIndex < frame.mvpMapCurves.size(); ++curveIndex)
        {
            MapCurve *pMapCurve = frame.mvpMapCurves[curveIndex];
            if (pMapCurve && curveIndex < frame.mvBezierCurves.size() && frame.mvBezierCurves[curveIndex].controlPoints.size() >= 2)
                observedCurvesByMapCurve[pMapCurve].push_back(curveIndex);
        }

        const double confidenceRadius = std::max(1.0f, mCurveConfig->matchSearchRadius);
        const double inverseConfidenceVariance = 1.0 / (2.0 * confidenceRadius * confidenceRadius);
        for (const auto &association : observedCurvesByMapCurve)
        {
            MapCurve *pMapCurve = association.first;
            if (!pMapCurve || pMapCurve->isBad())
                continue;

            const std::vector<cv::Point3d> worldPoints = pMapCurve->GetCurvePoints();
            std::vector<CurveSampleCorrespondence> curveCorrespondences;
            std::vector<cv::Point2f> projectedPoints;
            curveCorrespondences.reserve(worldPoints.size());
            projectedPoints.reserve(worldPoints.size());

            for (const cv::Point3d &worldPoint : worldPoints)
            {
                cv::Point2f projectedPoint;
                if (!ProjectWorldPoint(worldPoint, frame, projectedPoint) || projectedPoint.x < frame.mnMinX || projectedPoint.x > frame.mnMaxX || projectedPoint.y < frame.mnMinY || projectedPoint.y > frame.mnMaxY)
                    continue;

                bool foundClosestPoint = false;
                size_t bestCurveIndex = 0;
                double bestParameter = 0.0;
                double bestDistance = std::numeric_limits<double>::max();
                cv::Point2d bestPoint;
                cv::Point2d bestNormal;
                for (const size_t curveIndex : association.second)
                {
                    double parameter = 0.0;
                    double distance = 0.0;
                    cv::Point2d closestPoint;
                    cv::Point2d normal;
                    if (!frame.mvBezierCurves[curveIndex].FindClosestPoint(cv::Point2d(projectedPoint.x, projectedPoint.y), parameter, closestPoint, normal, distance))
                        continue;
                    if (distance < bestDistance)
                    {
                        foundClosestPoint = true;
                        bestCurveIndex = curveIndex;
                        bestParameter = parameter;
                        bestDistance = distance;
                        bestPoint = closestPoint;
                        bestNormal = normal;
                    }
                }
                if (!foundClosestPoint)
                    continue;

                CurveSampleCorrespondence correspondence;
                correspondence.pMapCurve = pMapCurve;
                correspondence.worldPoint = worldPoint;
                correspondence.observedPoint = bestPoint;
                correspondence.observedNormal = bestNormal;
                correspondence.observedCurveIndex = bestCurveIndex;
                correspondence.observedParameter = bestParameter;
                correspondence.initialDistance = bestDistance;
                curveCorrespondences.push_back(correspondence);
                projectedPoints.push_back(projectedPoint);
            }

            if (curveCorrespondences.empty())
                continue;

            std::vector<double> arcLengthWeights(curveCorrespondences.size(), 1.0);
            if (curveCorrespondences.size() > 1)
            {
                for (size_t sampleIndex = 0; sampleIndex < curveCorrespondences.size(); ++sampleIndex)
                {
                    const double previousLength = sampleIndex > 0 ? cv::norm(projectedPoints[sampleIndex] - projectedPoints[sampleIndex - 1]) : cv::norm(projectedPoints[1] - projectedPoints[0]);
                    const double nextLength = sampleIndex + 1 < projectedPoints.size() ? cv::norm(projectedPoints[sampleIndex + 1] - projectedPoints[sampleIndex]) : cv::norm(projectedPoints.back() - projectedPoints[projectedPoints.size() - 2]);
                    arcLengthWeights[sampleIndex] = std::max(1.0, std::min(12.0, 0.5 * (previousLength + nextLength)));
                }
            }

            double totalArcLengthWeight = 0.0;
            for (const double weight : arcLengthWeights)
                totalArcLengthWeight += weight;
            for (size_t sampleIndex = 0; sampleIndex < curveCorrespondences.size(); ++sampleIndex)
            {
                CurveSampleCorrespondence &correspondence = curveCorrespondences[sampleIndex];
                double confidence = std::exp(-correspondence.initialDistance * correspondence.initialDistance * inverseConfidenceVariance);
                if (correspondence.observedParameter <= 0.02 || correspondence.observedParameter >= 0.98)
                    confidence *= 0.25;
                correspondence.normalizedWeight = arcLengthWeights[sampleIndex] * std::max(1e-3, confidence) / totalArcLengthWeight;
                frame.mvCurveSampleCorrespondences.push_back(correspondence);
            }
        }
    }

    int CurveMatcher::AssociateMapCurvesToFrame(const std::vector<MapCurve *> &mapCurves, Frame &currentFrame)
    {
        currentFrame.mvCurveSampleCorrespondences.clear();
        currentFrame.mvCurveMatchDiagnostics.clear();
        if (mapCurves.empty())
            return 0;

        const float searchRadius = mCurveConfig->matchSearchRadius;
        const int minimumHits = mCurveConfig->minCandidateHits;
        const float minimumCoverage = mCurveConfig->minCandidateCoverage;
        const double unmatchedCost = mCurveConfig->unmatchedCost;

        std::unordered_map<MapCurve *, size_t> diagnosticIndexByMapCurve;
        for (MapCurve *pMapCurve : mapCurves)
        {
            if (!pMapCurve || diagnosticIndexByMapCurve.count(pMapCurve) != 0)
                continue;

            CurveMatchDiagnostic diagnostic;
            diagnostic.pMapCurve = pMapCurve;
            diagnostic.failureType = "pending";
            diagnostic.failureDetail = "matching not completed";
            diagnosticIndexByMapCurve[pMapCurve] = currentFrame.mvCurveMatchDiagnostics.size();
            currentFrame.mvCurveMatchDiagnostics.push_back(diagnostic);
        }

        std::vector<int> diagnosticStages(currentFrame.mvCurveMatchDiagnostics.size(), -1);
        const auto setDiagnostic = [&](const size_t diagnosticIndex, const int stage, const std::string &type, const std::string &detail)
        {
            if (diagnosticIndex >= currentFrame.mvCurveMatchDiagnostics.size() || stage < diagnosticStages[diagnosticIndex])
                return;
            diagnosticStages[diagnosticIndex] = stage;
            currentFrame.mvCurveMatchDiagnostics[diagnosticIndex].failureType = type;
            currentFrame.mvCurveMatchDiagnostics[diagnosticIndex].failureDetail = detail;
        };

        const auto finalizeDiagnostics = [&]()
        {
            std::unordered_set<MapCurve *> matchedMapCurves;
            for (MapCurve *pMapCurve : currentFrame.mvpMapCurves)
            {
                if (pMapCurve)
                    matchedMapCurves.insert(pMapCurve);
            }

            for (size_t diagnosticIndex = 0; diagnosticIndex < currentFrame.mvCurveMatchDiagnostics.size(); ++diagnosticIndex)
            {
                CurveMatchDiagnostic &diagnostic = currentFrame.mvCurveMatchDiagnostics[diagnosticIndex];
                diagnostic.matched = matchedMapCurves.count(diagnostic.pMapCurve) != 0;
                if (diagnostic.matched)
                {
                    diagnostic.failureType.clear();
                    diagnostic.failureDetail.clear();
                }
                else if (diagnostic.failureType == "pending")
                {
                    diagnostic.failureType = "assignment";
                    diagnostic.failureDetail = "no current Bezier curve was assigned";
                }
            }
        };

        if (currentFrame.mvCurveMatchDiagnostics.empty())
            return 0;

        if (currentFrame.mvBezierCurves.empty())
        {
            for (size_t diagnosticIndex = 0; diagnosticIndex < currentFrame.mvCurveMatchDiagnostics.size(); ++diagnosticIndex)
                setDiagnostic(diagnosticIndex, 0, "observation", "Bezier curve count 0 < 1");
            finalizeDiagnostics();
            return 0;
        }

        if (currentFrame.mTcw.empty())
        {
            for (size_t diagnosticIndex = 0; diagnosticIndex < currentFrame.mvCurveMatchDiagnostics.size(); ++diagnosticIndex)
                setDiagnostic(diagnosticIndex, 0, "pose", "current pose is unavailable");
            finalizeDiagnostics();
            return 0;
        }

        // 1. Convert the samples of each observed Bezier curve to the common floating-point representation used by the matcher.
        std::vector<std::vector<cv::Point2f>> observedCurves(currentFrame.mvBezierCurves.size());
        for (size_t curveIndex = 0; curveIndex < currentFrame.mvBezierCurves.size(); ++curveIndex)
        {
            const std::vector<orderedEdgePoint> &sourceSamples = currentFrame.mvBezierCurves[curveIndex].sampledPoints;
            std::vector<cv::Point2f> &targetSamples = observedCurves[curveIndex];
            targetSamples.reserve(sourceSamples.size());
            for (const orderedEdgePoint &point : sourceSamples)
                targetSamples.push_back(ToPoint2f(point));
        }

        // 2. Project every valid map curve into the current frame. The input may contain the same map curve more than once, so keep only the first occurrence.
        std::vector<ProjectedMapCurve> projectedCurves;
        std::vector<size_t> projectedDiagnosticIndices;
        projectedCurves.reserve(mapCurves.size());
        projectedDiagnosticIndices.reserve(mapCurves.size());
        std::unordered_set<MapCurve *> processedMapCurves;
        for (MapCurve *pMapCurve : mapCurves)
        {
            if (!pMapCurve || !processedMapCurves.insert(pMapCurve).second)
            {
                continue;
            }

            const size_t diagnosticIndex = diagnosticIndexByMapCurve[pMapCurve];
            if (pMapCurve->isBad())
            {
                setDiagnostic(diagnosticIndex, 0, "invalid", "MapCurve is marked bad");
                continue;
            }

            ProjectedMapCurve projectedCurve;
            if (ProjectMapCurve(pMapCurve, currentFrame, searchRadius, projectedCurve))
            {
                projectedCurves.push_back(std::move(projectedCurve));
                projectedDiagnosticIndices.push_back(diagnosticIndex);
                setDiagnostic(diagnosticIndex, 1, "candidate", "no nearby Bezier candidate");
            }
            else
                setDiagnostic(diagnosticIndex, 0, "projection", "visible projected samples < 2");
        }

        if (projectedCurves.empty())
        {
            finalizeDiagnostics();
            return 0;
        }

        // 3. Index all observed samples by image position. For each projected map sample, query the surrounding cells and then apply the exact circular radius test.
        CurveSampleGrid sampleGrid(currentFrame.mnMinX, currentFrame.mnMinY, currentFrame.mnMaxX, currentFrame.mnMaxY, searchRadius);
        for (size_t curveIndex = 0; curveIndex < observedCurves.size(); ++curveIndex)
        {
            for (size_t sampleIndex = 0; sampleIndex < observedCurves[curveIndex].size(); ++sampleIndex)
            {
                sampleGrid.Insert(observedCurves[curveIndex][sampleIndex], CurveSampleRef{curveIndex, sampleIndex});
            }
        }

        const float searchRadiusSquared = searchRadius * searchRadius;
        std::vector<std::vector<size_t>> candidatesByMap(projectedCurves.size());
        std::vector<CurveSampleRef> nearbySamples;

        for (size_t mapIndex = 0; mapIndex < projectedCurves.size(); ++mapIndex)
        {
            const std::vector<cv::Point2f> &mapSamples = projectedCurves[mapIndex].samples;
            std::vector<CandidateStatistics> candidateStatistics(observedCurves.size());
            const size_t diagnosticIndex = projectedDiagnosticIndices[mapIndex];

            for (const cv::Point2f &mapPoint : mapSamples)
            {
                sampleGrid.Query(mapPoint, searchRadius, nearbySamples);

                // A projected sample contributes at most one hit to a given observed curve. Retain its nearest sample.
                std::unordered_map<size_t, std::pair<size_t, float>> nearestHitByCurve;
                for (const CurveSampleRef &sample : nearbySamples)
                {
                    const cv::Point2f delta = observedCurves[sample.curveIndex][sample.sampleIndex] - mapPoint;
                    const float distanceSquared = delta.dot(delta);
                    if (distanceSquared > searchRadiusSquared)
                        continue;

                    const auto existing = nearestHitByCurve.find(sample.curveIndex);
                    if (existing == nearestHitByCurve.end() || distanceSquared < existing->second.second)
                    {
                        nearestHitByCurve[sample.curveIndex] = std::make_pair(sample.sampleIndex, distanceSquared);
                    }
                }

                for (const auto &hit : nearestHitByCurve)
                {
                    candidateStatistics[hit.first].AddHit(hit.second.first);
                }
            }

            size_t bestUniqueHitCount = 0;
            size_t bestRequiredHitCount = static_cast<size_t>(minimumHits);
            float bestCoverage = 0.0f;
            float bestCoverageAfterHitThreshold = 0.0f;
            bool hasEligibleObservedCurve = false;
            bool passedHitThreshold = false;
            bool hasBestHitStatistics = false;

            // Require both an absolute number of hits and coverage of the shorter curve. This prevents a single accidental intersection from becoming a matching candidate.
            for (size_t curveIndex = 0; curveIndex < observedCurves.size(); ++curveIndex)
            {
                const size_t shorterSampleCount = std::min(mapSamples.size(), observedCurves[curveIndex].size());
                if (shorterSampleCount < 2)
                    continue;
                hasEligibleObservedCurve = true;

                const CandidateStatistics &statistics = candidateStatistics[curveIndex];
                const size_t uniqueHitCount = std::min(statistics.matchedMapSamples, statistics.matchedObservedSamples.size());
                const size_t requiredHitCount = std::min(shorterSampleCount, static_cast<size_t>(minimumHits));
                const float coverage = static_cast<float>(uniqueHitCount) / static_cast<float>(shorterSampleCount);
                if (!hasBestHitStatistics || uniqueHitCount > bestUniqueHitCount ||
                    (uniqueHitCount == bestUniqueHitCount && coverage > bestCoverage))
                {
                    hasBestHitStatistics = true;
                    bestUniqueHitCount = uniqueHitCount;
                    bestRequiredHitCount = requiredHitCount;
                    bestCoverage = coverage;
                }
                if (uniqueHitCount >= requiredHitCount)
                {
                    passedHitThreshold = true;
                    bestCoverageAfterHitThreshold = std::max(bestCoverageAfterHitThreshold, coverage);
                }

                if (uniqueHitCount >= requiredHitCount && coverage >= minimumCoverage)
                {
                    candidatesByMap[mapIndex].push_back(curveIndex);
                }
            }

            if (candidatesByMap[mapIndex].empty())
            {
                if (!hasEligibleObservedCurve)
                    setDiagnostic(diagnosticIndex, 1, "samples", "observed sample count < 2");
                else if (!passedHitThreshold)
                    setDiagnostic(diagnosticIndex, 1, "hits", "hits " + std::to_string(bestUniqueHitCount) + " < " + std::to_string(bestRequiredHitCount));
                else
                    setDiagnostic(diagnosticIndex, 1, "coverage", "coverage " + DiagnosticNumber(bestCoverageAfterHitThreshold) + " < " + DiagnosticNumber(minimumCoverage));
            }
            else
                setDiagnostic(diagnosticIndex, 2, "similarity", "no candidate passed the shape thresholds");
        }

        // 4. Create one assignment row for each currently unassociated observation. Similarity is evaluated only for candidates found by the radius search above.
        std::vector<size_t> rowToCurveIndex;
        std::vector<int> curveToRow(observedCurves.size(), -1);
        for (size_t curveIndex = 0; curveIndex < observedCurves.size(); ++curveIndex)
        {
            if (observedCurves[curveIndex].size() < 2 || currentFrame.mvpMapCurves[curveIndex])
            {
                continue;
            }

            curveToRow[curveIndex] = static_cast<int>(rowToCurveIndex.size());
            rowToCurveIndex.push_back(curveIndex);
        }

        if (rowToCurveIndex.empty())
        {
            for (size_t mapIndex = 0; mapIndex < projectedCurves.size(); ++mapIndex)
            {
                if (!candidatesByMap[mapIndex].empty())
                    setDiagnostic(projectedDiagnosticIndices[mapIndex], 4, "assignment", "all Bezier observations are already associated");
            }
            BuildFixedSampleCorrespondences(currentFrame);
            finalizeDiagnostics();
            return 0;
        }

        const size_t rowCount = rowToCurveIndex.size();
        const size_t mapCount = projectedCurves.size();
        std::vector<std::vector<CurveSimilarity>> similarities(rowCount, std::vector<CurveSimilarity>(mapCount));
        std::vector<std::vector<double>> costMatrix(rowCount, std::vector<double>(mapCount + rowCount, kInvalidCost));
        std::vector<bool> evaluatedSimilarity(mapCount, false);
        std::vector<bool> hasValidSimilarity(mapCount, false);
        std::vector<double> bestValidCost(mapCount, kInvalidCost);
        std::vector<double> bestInvalidScore(mapCount, kInvalidCost);
        std::vector<std::string> bestInvalidType(mapCount);
        std::vector<std::string> bestInvalidDetail(mapCount);

        for (size_t mapIndex = 0; mapIndex < mapCount; ++mapIndex)
        {
            for (const size_t curveIndex : candidatesByMap[mapIndex])
            {
                const int row = curveToRow[curveIndex];
                if (row < 0)
                    continue;

                CurveSimilarity similarity = ComputeCurveSimilarity(projectedCurves[mapIndex], observedCurves[curveIndex], searchRadius);
                similarities[static_cast<size_t>(row)][mapIndex] = similarity;
                evaluatedSimilarity[mapIndex] = true;
                if (similarity.valid)
                {
                    costMatrix[static_cast<size_t>(row)][mapIndex] = similarity.cost;
                    hasValidSimilarity[mapIndex] = true;
                    bestValidCost[mapIndex] = std::min(bestValidCost[mapIndex], similarity.cost);
                }
                else
                {
                    const bool distanceFailed = std::isfinite(similarity.meanDistance) && similarity.meanDistance > searchRadius;
                    const bool directionFailed = std::isfinite(similarity.meanDirectionCost) && similarity.meanDirectionCost > kMaxMeanDirectionCost;
                    const double distanceRatio = distanceFailed ? similarity.meanDistance / searchRadius : 1.0;
                    const double directionRatio = directionFailed ? similarity.meanDirectionCost / kMaxMeanDirectionCost : 1.0;
                    const double failureScore = std::max(distanceRatio, directionRatio);
                    if (failureScore < bestInvalidScore[mapIndex])
                    {
                        bestInvalidScore[mapIndex] = failureScore;
                        if (distanceFailed && directionFailed)
                        {
                            bestInvalidType[mapIndex] = "distance+direction";
                            bestInvalidDetail[mapIndex] =
                                "distance " + DiagnosticNumber(similarity.meanDistance) + " > " + DiagnosticNumber(searchRadius) +
                                ", direction " + DiagnosticNumber(similarity.meanDirectionCost) + " > " + DiagnosticNumber(kMaxMeanDirectionCost);
                        }
                        else if (distanceFailed)
                        {
                            bestInvalidType[mapIndex] = "distance";
                            bestInvalidDetail[mapIndex] = "distance " + DiagnosticNumber(similarity.meanDistance) + " > " + DiagnosticNumber(searchRadius) + " px";
                        }
                        else if (directionFailed)
                        {
                            bestInvalidType[mapIndex] = "direction";
                            bestInvalidDetail[mapIndex] = "direction " + DiagnosticNumber(similarity.meanDirectionCost) + " > " + DiagnosticNumber(kMaxMeanDirectionCost);
                        }
                        else
                        {
                            bestInvalidType[mapIndex] = "similarity";
                            bestInvalidDetail[mapIndex] = "shape metrics are invalid";
                        }
                    }
                }
            }

            const size_t diagnosticIndex = projectedDiagnosticIndices[mapIndex];
            if (!candidatesByMap[mapIndex].empty())
            {
                if (!evaluatedSimilarity[mapIndex])
                    setDiagnostic(diagnosticIndex, 2, "assignment", "candidate Bezier curves are already associated");
                else if (!hasValidSimilarity[mapIndex])
                    setDiagnostic(diagnosticIndex, 2,
                                  bestInvalidType[mapIndex].empty() ? "similarity" : bestInvalidType[mapIndex],
                                  bestInvalidDetail[mapIndex].empty() ? "no candidate passed the shape thresholds" : bestInvalidDetail[mapIndex]);
                else if (bestValidCost[mapIndex] >= unmatchedCost)
                    setDiagnostic(diagnosticIndex, 3, "cost", "cost " + DiagnosticNumber(bestValidCost[mapIndex]) + " >= " + DiagnosticNumber(unmatchedCost));
                else
                    setDiagnostic(diagnosticIndex, 4, "assignment", "a competing MapCurve won the global assignment");
            }
        }

        // Give every observation its own dummy column, allowing the Hungarian solver to leave it unmatched.
        for (size_t row = 0; row < rowCount; ++row)
            costMatrix[row][mapCount + row] = unmatchedCost;

        HungarianAlgorithm hungarian;
        std::vector<int> assignment;
        hungarian.Solve(costMatrix, assignment);

        // 5. Accept the one-to-one anchors selected by the global assignment. Reject an ambiguous result when another map candidate has a nearly identical cost.
        int matchCount = 0;
        std::vector<bool> matchedRows(rowCount, false);
        std::vector<bool> anchoredMaps(mapCount, false);
        std::vector<std::vector<AcceptedFragment>> acceptedFragments(mapCount);

        for (size_t row = 0; row < rowCount; ++row)
        {
            if (row >= assignment.size() || assignment[row] < 0 || static_cast<size_t>(assignment[row]) >= mapCount)
            {
                continue;
            }

            const size_t mapIndex = static_cast<size_t>(assignment[row]);
            const CurveSimilarity &chosen = similarities[row][mapIndex];
            if (!chosen.valid || chosen.cost >= unmatchedCost)
                continue;

            double alternativeCost = kInvalidCost;
            size_t alternativeMapIndex = mapCount;
            for (size_t alternativeMap = 0; alternativeMap < mapCount; ++alternativeMap)
            {
                if (alternativeMap == mapIndex)
                    continue;
                const CurveSimilarity &alternative = similarities[row][alternativeMap];
                if (alternative.valid && alternative.cost < alternativeCost)
                {
                    alternativeCost = alternative.cost;
                    alternativeMapIndex = alternativeMap;
                }
            }
            if (alternativeMapIndex < mapCount && chosen.cost > alternativeCost)
            {
                MapCurve *pAlternativeMapCurve = projectedCurves[alternativeMapIndex].pMapCurve;
                setDiagnostic(projectedDiagnosticIndices[mapIndex], 5, "ambiguity",
                              "cost " + DiagnosticNumber(chosen.cost) +
                                  " > alternative MapCurve ID " +
                                  (pAlternativeMapCurve ? std::to_string(pAlternativeMapCurve->mnId) : std::string("unknown")) +
                                  " cost " + DiagnosticNumber(alternativeCost));
                continue;
            }

            const size_t curveIndex = rowToCurveIndex[row];
            currentFrame.mvpMapCurves[curveIndex] = projectedCurves[mapIndex].pMapCurve;
            matchedRows[row] = true;
            anchoredMaps[mapIndex] = true;
            acceptedFragments[mapIndex].push_back(AcceptedFragment{curveIndex, chosen});
            ++matchCount;
        }

        // 6. A map curve may appear as several Bezier fragments in the current frame. Attach an unmatched fragment to an anchored map curve when their projected spans and endpoints are compatible.
        for (size_t row = 0; row < rowCount; ++row)
        {
            if (matchedRows[row])
                continue;

            const size_t curveIndex = rowToCurveIndex[row];
            size_t bestMapIndex = mapCount;
            double bestCost = kInvalidCost;
            double secondBestCost = kInvalidCost;

            for (size_t mapIndex = 0; mapIndex < mapCount; ++mapIndex)
            {
                if (!anchoredMaps[mapIndex])
                    continue;

                const CurveSimilarity &similarity = similarities[row][mapIndex];
                if (!similarity.valid || similarity.cost >= unmatchedCost || !FragmentIsCompatible(curveIndex, similarity, acceptedFragments[mapIndex], currentFrame, observedCurves, projectedCurves[mapIndex].samples.size()))
                {
                    continue;
                }

                if (similarity.cost < bestCost)
                {
                    secondBestCost = bestCost;
                    bestCost = similarity.cost;
                    bestMapIndex = mapIndex;
                }
                else if (similarity.cost < secondBestCost)
                {
                    secondBestCost = similarity.cost;
                }
            }

            if (bestMapIndex == mapCount || (secondBestCost < kInvalidCost && bestCost > 0.90 * secondBestCost))
            {
                continue;
            }

            currentFrame.mvpMapCurves[curveIndex] = projectedCurves[bestMapIndex].pMapCurve;
            acceptedFragments[bestMapIndex].push_back(AcceptedFragment{curveIndex, similarities[row][bestMapIndex]});
            ++matchCount;
        }

        BuildFixedSampleCorrespondences(currentFrame);
        finalizeDiagnostics();
        return matchCount;
    }

    int CurveMatcher::Fuse(KeyFrame *pKF, const std::vector<MapCurve *> &vpMapCurves)
    {
        if (!mCurveConfig || !pKF || pKF->isBad() ||
            pKF->mvBezierCurves.empty() || vpMapCurves.empty())
            return 0;

        const float searchRadius = std::max(1.0f, mCurveConfig->matchSearchRadius);
        const float searchRadiusSquared = searchRadius * searchRadius;
        const int minimumHits = std::max(1, mCurveConfig->minCandidateHits);
        const float minimumCoverage = std::max(0.0f, std::min(1.0f, mCurveConfig->minCandidateCoverage));
        const double unmatchedCost = std::max(0.1f, mCurveConfig->unmatchedCost);

        std::vector<std::vector<cv::Point2f>> observedCurves(pKF->mvBezierCurves.size());
        std::vector<size_t> rowToCurveIndex;
        for (size_t curveIndex = 0; curveIndex < pKF->mvBezierCurves.size(); ++curveIndex)
        {
            const std::vector<orderedEdgePoint> &sourceSamples = pKF->mvBezierCurves[curveIndex].sampledPoints;
            std::vector<cv::Point2f> &targetSamples = observedCurves[curveIndex];
            targetSamples.reserve(sourceSamples.size());
            for (const orderedEdgePoint &point : sourceSamples)
                targetSamples.push_back(ToPoint2f(point));
            if (targetSamples.size() >= 2)
                rowToCurveIndex.push_back(curveIndex);
        }
        if (rowToCurveIndex.empty())
            return 0;

        std::vector<ProjectedMapCurve> projectedCurves;
        std::unordered_set<MapCurve *> processedMapCurves;
        projectedCurves.reserve(vpMapCurves.size());
        for (MapCurve *pMapCurve : vpMapCurves)
        {
            if (!pMapCurve || !processedMapCurves.insert(pMapCurve).second || pMapCurve->isBad())
                continue;
            ProjectedMapCurve projected;
            if (ProjectMapCurve(pMapCurve, pKF, searchRadius, projected))
                projectedCurves.push_back(std::move(projected));
        }
        if (projectedCurves.empty())
            return 0;

        const size_t rowCount = rowToCurveIndex.size();
        const size_t mapCount = projectedCurves.size();
        std::vector<std::vector<CurveSimilarity>> similarities(rowCount, std::vector<CurveSimilarity>(mapCount));
        std::vector<std::vector<double>> costMatrix(rowCount, std::vector<double>(mapCount + rowCount, kInvalidCost));

        for (size_t row = 0; row < rowCount; ++row)
        {
            const size_t curveIndex = rowToCurveIndex[row];
            const std::vector<cv::Point2f> &observedCurve = observedCurves[curveIndex];
            for (size_t mapIndex = 0; mapIndex < mapCount; ++mapIndex)
            {
                const std::vector<cv::Point2f> &mapSamples = projectedCurves[mapIndex].samples;
                const size_t shorterSampleCount = std::min(mapSamples.size(), observedCurve.size());
                if (shorterSampleCount < 2)
                    continue;

                size_t matchedMapSamples = 0;
                std::unordered_set<size_t> matchedObservedSamples;
                for (const cv::Point2f &mapPoint : mapSamples)
                {
                    size_t bestObservedIndex = observedCurve.size();
                    float bestDistanceSquared = searchRadiusSquared;
                    for (size_t observedIndex = 0; observedIndex < observedCurve.size(); ++observedIndex)
                    {
                        const cv::Point2f difference = observedCurve[observedIndex] - mapPoint;
                        const float distanceSquared = difference.dot(difference);
                        if (distanceSquared <= bestDistanceSquared)
                        {
                            bestDistanceSquared = distanceSquared;
                            bestObservedIndex = observedIndex;
                        }
                    }
                    if (bestObservedIndex < observedCurve.size())
                    {
                        ++matchedMapSamples;
                        matchedObservedSamples.insert(bestObservedIndex);
                    }
                }

                const size_t uniqueHitCount = std::min(matchedMapSamples, matchedObservedSamples.size());
                const size_t requiredHitCount = std::min(shorterSampleCount, static_cast<size_t>(minimumHits));
                const float coverage = static_cast<float>(uniqueHitCount) / static_cast<float>(shorterSampleCount);
                if (uniqueHitCount < requiredHitCount || coverage < minimumCoverage)
                    continue;

                CurveSimilarity similarity = ComputeCurveSimilarity(projectedCurves[mapIndex], observedCurve, searchRadius);
                similarities[row][mapIndex] = similarity;
                if (similarity.valid && similarity.cost < unmatchedCost)
                    costMatrix[row][mapIndex] = similarity.cost;
            }
            costMatrix[row][mapCount + row] = unmatchedCost;
        }

        HungarianAlgorithm hungarian;
        std::vector<int> assignment;
        hungarian.Solve(costMatrix, assignment);

        int fusedCount = 0;
        for (size_t row = 0; row < rowCount; ++row)
        {
            if (row >= assignment.size() || assignment[row] < 0 || static_cast<size_t>(assignment[row]) >= mapCount)
                continue;
            const size_t mapIndex = static_cast<size_t>(assignment[row]);
            const CurveSimilarity &chosen = similarities[row][mapIndex];
            if (!chosen.valid || chosen.cost >= unmatchedCost)
                continue;

            double alternativeCost = kInvalidCost;
            for (size_t alternativeMapIndex = 0; alternativeMapIndex < mapCount; ++alternativeMapIndex)
            {
                if (alternativeMapIndex == mapIndex)
                    continue;
                const CurveSimilarity &alternative = similarities[row][alternativeMapIndex];
                if (alternative.valid && alternative.cost < alternativeCost)
                    alternativeCost = alternative.cost;
            }
            if (alternativeCost < kInvalidCost && chosen.cost > 0.90 * alternativeCost)
                continue;

            const size_t curveIndex = rowToCurveIndex[row];
            MapCurve *pCandidate = projectedCurves[mapIndex].pMapCurve;
            if (!pCandidate || pCandidate->isBad())
                continue;

            MapCurve *pExisting = pKF->GetMapCurve(curveIndex);
            if (pExisting && pExisting->isBad())
            {
                pKF->ReplaceMapCurveMatch(
                    curveIndex, pExisting, NULL);
                pExisting = static_cast<MapCurve *>(NULL);
            }

            if (!pExisting)
            {
                if (pCandidate->AssociateWithKeyFrame(pKF, curveIndex))
                    ++fusedCount;
                continue;
            }

            if (pExisting == pCandidate)
            {
                pCandidate->AssociateWithKeyFrame(pKF, curveIndex);
                continue;
            }

            MapCurve *pSurvivor = pExisting;
            MapCurve *pDiscarded = pCandidate;
            const int existingObservations = pExisting->Observations();
            const int candidateObservations = pCandidate->Observations();
            const unsigned long existingWeight = pExisting->GetTotalFusionWeight();
            const unsigned long candidateWeight = pCandidate->GetTotalFusionWeight();
            if (candidateObservations > existingObservations || (candidateObservations == existingObservations && candidateWeight > existingWeight) || (candidateObservations == existingObservations && candidateWeight == existingWeight && pCandidate->mnId < pExisting->mnId))
            {
                pSurvivor = pCandidate;
                pDiscarded = pExisting;
            }

            if (pDiscarded->Replace(pSurvivor, mCurveConfig))
                ++fusedCount;
        }

        if (fusedCount > 0)
            pKF->UpdateConnections();
        return fusedCount;
    }
} // namespace ORB_SLAM2
