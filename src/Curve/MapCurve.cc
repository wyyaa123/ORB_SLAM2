#include "Curve/MapCurve.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

namespace ORB_SLAM2
{
    long unsigned int MapCurve::nNextId = 0;
    std::mutex MapCurve::mGlobalMutex;

    MapCurve::MapCurve(std::vector<cv::Point3d> curvePoints, KeyFrame *pReferenceKF, Map *pMap) : mnFirstKFid(pReferenceKF->mnId), mnFirstFrame(pReferenceKF->mnFrameId), nObs(0), mbTrackInView(false), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mpRefKF(pReferenceKF), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
    {
        mCurvePoints = std::move(curvePoints);
        mCurvePointFusionWeights.assign(mCurvePoints.size(), 1);

        // MapCurves can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    // 在关键帧pKF中的第idx个曲线特征
    void MapCurve::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        std::vector<size_t> &indices = mObservations[pKF];
        if (std::find(indices.begin(), indices.end(), idx) != indices.end())
            return;
        if (indices.empty())
            ++nObs;
        indices.push_back(idx);
        std::sort(indices.begin(), indices.end());
    }

    void MapCurve::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF))
            {
                nObs--;

                mObservations.erase(pKF);
                if (mpRefKF == pKF)
                    mpRefKF = mObservations.empty() ? static_cast<KeyFrame *>(NULL) : mObservations.begin()->first;

                if (nObs < 2)
                    bBad = true;
            }
        }

        if (bBad)
            SetBadFlag();
    }

    MapCurve *MapCurve::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapCurve::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapCurve::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapCurve::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    std::vector<cv::Point3d> MapCurve::GetCurvePoints()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mCurvePoints;
    }

    bool MapCurve::ProjectWorldPoint(const cv::Point3d &worldPoint, const Frame &frame, cv::Point2d &imagePoint)
    {
        if (frame.mTcw.empty())
            return false;

        const double cameraX = frame.mTcw.at<float>(0, 0) * worldPoint.x + frame.mTcw.at<float>(0, 1) * worldPoint.y + frame.mTcw.at<float>(0, 2) * worldPoint.z + frame.mTcw.at<float>(0, 3);
        const double cameraY = frame.mTcw.at<float>(1, 0) * worldPoint.x + frame.mTcw.at<float>(1, 1) * worldPoint.y + frame.mTcw.at<float>(1, 2) * worldPoint.z + frame.mTcw.at<float>(1, 3);
        const double cameraZ = frame.mTcw.at<float>(2, 0) * worldPoint.x + frame.mTcw.at<float>(2, 1) * worldPoint.y + frame.mTcw.at<float>(2, 2) * worldPoint.z + frame.mTcw.at<float>(2, 3);
        if (cameraZ <= 0.0)
            return false;

        imagePoint.x = frame.fx * cameraX / cameraZ + frame.cx;
        imagePoint.y = frame.fy * cameraY / cameraZ + frame.cy;
        return imagePoint.x >= frame.mnMinX && imagePoint.x <= frame.mnMaxX && imagePoint.y >= frame.mnMinY && imagePoint.y <= frame.mnMaxY;
    }

    size_t MapCurve::FindNearestPointIndex(const cv::Point2d &queryPoint, const std::vector<cv::Point2d> &points)
    {
        if (points.empty())
            return 0;

        size_t nearestIndex = 0;
        double nearestDistanceSquared = std::numeric_limits<double>::max();
        for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
        {
            const cv::Point2d difference = points[pointIndex] - queryPoint;
            const double distanceSquared = difference.dot(difference);
            if (distanceSquared < nearestDistanceSquared)
            {
                nearestDistanceSquared = distanceSquared;
                nearestIndex = pointIndex;
            }
        }
        return nearestIndex;
    }

    void MapCurve::SmoothGeometry(const CurveConfigPtr &curveConfig)
    {
        if (!curveConfig || curveConfig->geometrySmoothingWeight <= 0.0f || mCurvePoints.size() < 3)
            return;

        const double smoothingWeight = curveConfig->geometrySmoothingWeight;
        std::vector<cv::Point3d> smoothedPoints = mCurvePoints;
        for (size_t pointIndex = 1; pointIndex + 1 < mCurvePoints.size(); ++pointIndex)
        {
            const cv::Point3d &previousPoint = mCurvePoints[pointIndex - 1];
            const cv::Point3d &currentPoint = mCurvePoints[pointIndex];
            const cv::Point3d &nextPoint = mCurvePoints[pointIndex + 1];
            if (cv::norm(currentPoint - previousPoint) > curveConfig->extensionMax3DGap || cv::norm(nextPoint - currentPoint) > curveConfig->extensionMax3DGap)
                continue;

            const cv::Point3d localMidpoint = (previousPoint + nextPoint) * 0.5;
            const cv::Point3d correction = localMidpoint - currentPoint;
            if (cv::norm(correction) > curveConfig->fusionMax3DDistance)
                continue;

            smoothedPoints[pointIndex] = currentPoint + correction * smoothingWeight;
        }
        mCurvePoints.swap(smoothedPoints);
    }

    size_t MapCurve::ExtendWithObservation(const Frame &frame, const size_t observedCurveIndex, const CurveConfigPtr &curveConfig)
    {
        if (!curveConfig || observedCurveIndex >= frame.mvBezierCurves.size() || frame.mTcw.empty())
            return 0;

        const BezierCurve &observedCurve = frame.mvBezierCurves[observedCurveIndex];
        const std::vector<cv::Point3d> observedWorldPoints = frame.UnprojectCurve(static_cast<int>(observedCurveIndex));
        std::vector<cv::Point2d> observedImagePoints;
        observedImagePoints.reserve(observedCurve.sampledPoints.size());
        for (const orderedEdgePoint &point : observedCurve.sampledPoints)
        {
            if (point.depth > 0.0f)
                observedImagePoints.push_back(cv::Point2d(point.x, point.y));
        }
        if (observedWorldPoints.size() < 2 || observedWorldPoints.size() != observedImagePoints.size())
            return 0;

        unique_lock<mutex> lock(mMutexPos);
        if (mCurvePoints.size() < 2)
        {
            mCurvePoints = observedWorldPoints;
            mCurvePointFusionWeights.assign(mCurvePoints.size(), 1);
            return observedWorldPoints.size();
        }
        if (mCurvePointFusionWeights.size() != mCurvePoints.size())
            mCurvePointFusionWeights.assign(mCurvePoints.size(), 1);

        std::vector<cv::Point2d> projectedMapPoints;
        std::vector<size_t> projectedMapIndices;
        projectedMapPoints.reserve(mCurvePoints.size());
        projectedMapIndices.reserve(mCurvePoints.size());
        for (size_t mapPointIndex = 0; mapPointIndex < mCurvePoints.size(); ++mapPointIndex)
        {
            cv::Point2d projectedPoint;
            if (!ProjectWorldPoint(mCurvePoints[mapPointIndex], frame, projectedPoint) || projectedPoint.x < frame.mnMinX || projectedPoint.x > frame.mnMaxX || projectedPoint.y < frame.mnMinY || projectedPoint.y > frame.mnMaxY)
                continue;
            projectedMapPoints.push_back(projectedPoint);
            projectedMapIndices.push_back(mapPointIndex);
        }
        if (projectedMapPoints.size() < 2)
            return 0;

        std::vector<cv::Point2d> orientedImagePoints = observedImagePoints;
        std::vector<cv::Point3d> orientedWorldPoints = observedWorldPoints;
        // Make the current observation follow the same front-to-back order as the projected map curve.
        const double forwardEndpointCost = cv::norm(orientedImagePoints.front() - projectedMapPoints.front()) + cv::norm(orientedImagePoints.back() - projectedMapPoints.back());
        const double reverseEndpointCost = cv::norm(orientedImagePoints.front() - projectedMapPoints.back()) + cv::norm(orientedImagePoints.back() - projectedMapPoints.front());
        if (reverseEndpointCost < forwardEndpointCost)
        {
            std::reverse(orientedImagePoints.begin(), orientedImagePoints.end());
            std::reverse(orientedWorldPoints.begin(), orientedWorldPoints.end());
        }

        std::vector<size_t> nearestObservedIndices(projectedMapPoints.size(), 0);
        size_t firstOverlapIndex = orientedImagePoints.size();
        size_t lastOverlapIndex = 0;
        for (size_t projectedPointIndex = 0; projectedPointIndex < projectedMapPoints.size(); ++projectedPointIndex)
        {
            const size_t nearestObservedIndex = FindNearestPointIndex(projectedMapPoints[projectedPointIndex], orientedImagePoints);
            nearestObservedIndices[projectedPointIndex] = nearestObservedIndex;
            firstOverlapIndex = std::min(firstOverlapIndex, nearestObservedIndex);
            lastOverlapIndex = std::max(lastOverlapIndex, nearestObservedIndex);
        }
        if (firstOverlapIndex >= orientedImagePoints.size())
            return 0;

        size_t fusedPointCount = 0;
        bool hasPreviousFusedObservation = false;
        size_t previousFusedObservationIndex = 0;
        for (size_t projectedPointIndex = 0; projectedPointIndex < projectedMapPoints.size(); ++projectedPointIndex)
        {
            const size_t observedPointIndex = nearestObservedIndices[projectedPointIndex];
            if (hasPreviousFusedObservation && observedPointIndex <= previousFusedObservationIndex)
                continue;

            const size_t mapPointIndex = projectedMapIndices[projectedPointIndex];
            const double pixelDistance = cv::norm(projectedMapPoints[projectedPointIndex] - orientedImagePoints[observedPointIndex]);
            const double spatialDistance = cv::norm(mCurvePoints[mapPointIndex] - orientedWorldPoints[observedPointIndex]);
            if (pixelDistance > curveConfig->fusionMaxPixelDistance || spatialDistance > curveConfig->fusionMax3DDistance)
                continue;

            const unsigned int fusionWeight = std::min(mCurvePointFusionWeights[mapPointIndex], static_cast<unsigned int>(curveConfig->maximumFusionWeight));
            const double observationWeight = 1.0 / static_cast<double>(fusionWeight + 1);
            mCurvePoints[mapPointIndex] = mCurvePoints[mapPointIndex] * (1.0 - observationWeight) + orientedWorldPoints[observedPointIndex] * observationWeight;
            if (mCurvePointFusionWeights[mapPointIndex] < static_cast<unsigned int>(curveConfig->maximumFusionWeight))
                ++mCurvePointFusionWeights[mapPointIndex];
            previousFusedObservationIndex = observedPointIndex;
            hasPreviousFusedObservation = true;
            ++fusedPointCount;
        }

        const bool mapFrontIsVisible = projectedMapIndices.front() == 0;
        const bool mapBackIsVisible = projectedMapIndices.back() + 1 == mCurvePoints.size();
        const size_t prefixCandidateCount = mapFrontIsVisible ? firstOverlapIndex : 0;
        const size_t suffixCandidateCount = mapBackIsVisible && lastOverlapIndex + 1 < orientedWorldPoints.size() ? orientedWorldPoints.size() - lastOverlapIndex - 1 : 0;

        const auto directionCosine = [](const cv::Point2d &firstStart, const cv::Point2d &firstEnd, const cv::Point2d &secondStart, const cv::Point2d &secondEnd)
        {
            const cv::Point2d firstDirection = firstEnd - firstStart;
            const cv::Point2d secondDirection = secondEnd - secondStart;
            const double denominator = cv::norm(firstDirection) * cv::norm(secondDirection);
            if (denominator <= std::numeric_limits<double>::epsilon())
                return -1.0;
            return firstDirection.dot(secondDirection) / denominator;
        };

        bool acceptPrefix = prefixCandidateCount >= static_cast<size_t>(curveConfig->minimumExtensionSamples);
        if (acceptPrefix)
        {
            const size_t lastPrefixIndex = firstOverlapIndex - 1;
            const double pixelGap = cv::norm(orientedImagePoints[lastPrefixIndex] - projectedMapPoints.front());
            const double spatialGap = cv::norm(orientedWorldPoints[lastPrefixIndex] - mCurvePoints.front());
            const double joinDirectionCosine = directionCosine(orientedImagePoints[lastPrefixIndex - 1], orientedImagePoints[lastPrefixIndex], projectedMapPoints[0], projectedMapPoints[1]);
            acceptPrefix = pixelGap <= curveConfig->extensionMaxPixelGap && spatialGap <= curveConfig->extensionMax3DGap && joinDirectionCosine >= curveConfig->minimumJoinDirectionCosine;
        }

        bool acceptSuffix = suffixCandidateCount >= static_cast<size_t>(curveConfig->minimumExtensionSamples);
        if (acceptSuffix)
        {
            const size_t firstSuffixIndex = lastOverlapIndex + 1;
            const double pixelGap = cv::norm(projectedMapPoints.back() - orientedImagePoints[firstSuffixIndex]);
            const double spatialGap = cv::norm(mCurvePoints.back() - orientedWorldPoints[firstSuffixIndex]);
            const double joinDirectionCosine = directionCosine(projectedMapPoints[projectedMapPoints.size() - 2], projectedMapPoints.back(), orientedImagePoints[firstSuffixIndex], orientedImagePoints[firstSuffixIndex + 1]);
            acceptSuffix = pixelGap <= curveConfig->extensionMaxPixelGap && spatialGap <= curveConfig->extensionMax3DGap && joinDirectionCosine >= curveConfig->minimumJoinDirectionCosine;
        }

        const size_t prefixCount = acceptPrefix ? prefixCandidateCount : 0;
        const size_t suffixCount = acceptSuffix ? suffixCandidateCount : 0;

        std::vector<cv::Point3d> extendedCurve;
        std::vector<unsigned int> extendedFusionWeights;
        extendedCurve.reserve(prefixCount + mCurvePoints.size() + suffixCount);
        extendedFusionWeights.reserve(prefixCount + mCurvePointFusionWeights.size() + suffixCount);
        if (prefixCount > 0)
        {
            extendedCurve.insert(extendedCurve.end(), orientedWorldPoints.begin(), orientedWorldPoints.begin() + firstOverlapIndex);
            extendedFusionWeights.insert(extendedFusionWeights.end(), prefixCount, 1);
        }
        extendedCurve.insert(extendedCurve.end(), mCurvePoints.begin(), mCurvePoints.end());
        extendedFusionWeights.insert(extendedFusionWeights.end(), mCurvePointFusionWeights.begin(), mCurvePointFusionWeights.end());
        if (suffixCount > 0)
        {
            extendedCurve.insert(extendedCurve.end(), orientedWorldPoints.begin() + lastOverlapIndex + 1, orientedWorldPoints.end());
            extendedFusionWeights.insert(extendedFusionWeights.end(), suffixCount, 1);
        }
        mCurvePoints.swap(extendedCurve);
        mCurvePointFusionWeights.swap(extendedFusionWeights);
        if (fusedPointCount > 0 || prefixCount > 0 || suffixCount > 0)
            SmoothGeometry(curveConfig);
        return prefixCount + suffixCount;
    }

    unsigned long MapCurve::GetTotalFusionWeight()
    {
        unique_lock<mutex> lock(mMutexPos);
        return std::accumulate(mCurvePointFusionWeights.begin(), mCurvePointFusionWeights.end(), 0UL);
    }

    bool MapCurve::MergeGeometryFrom(MapCurve *pOther, const CurveConfigPtr &curveConfig)
    {
        if (!pOther || pOther == this || !curveConfig || pOther->isBad())
            return false;

        std::vector<cv::Point3d> otherPoints;
        std::vector<unsigned int> otherWeights;
        {
            unique_lock<mutex> otherLock(pOther->mMutexPos);
            otherPoints = pOther->mCurvePoints;
            otherWeights = pOther->mCurvePointFusionWeights;
        }
        if (otherPoints.size() < 2)
            return false;
        if (otherWeights.size() != otherPoints.size())
            otherWeights.assign(otherPoints.size(), 1);

        unique_lock<mutex> lock(mMutexPos);
        if (mCurvePoints.size() < 2)
        {
            mCurvePoints = otherPoints;
            mCurvePointFusionWeights = otherWeights;
            return true;
        }
        if (mCurvePointFusionWeights.size() != mCurvePoints.size())
            mCurvePointFusionWeights.assign(mCurvePoints.size(), 1);

        const double forwardEndpointCost = cv::norm(otherPoints.front() - mCurvePoints.front()) + cv::norm(otherPoints.back() - mCurvePoints.back());
        const double reverseEndpointCost = cv::norm(otherPoints.front() - mCurvePoints.back()) + cv::norm(otherPoints.back() - mCurvePoints.front());
        if (reverseEndpointCost < forwardEndpointCost)
        {
            std::reverse(otherPoints.begin(), otherPoints.end());
            std::reverse(otherWeights.begin(), otherWeights.end());
        }

        struct GeometryMatch
        {
            size_t currentIndex;
            size_t otherIndex;
            double distance;
        };

        std::vector<GeometryMatch> matches;
        std::vector<double> distances;
        size_t nextCurrentIndex = 0;
        for (size_t otherIndex = 0; otherIndex < otherPoints.size() && nextCurrentIndex < mCurvePoints.size(); ++otherIndex)
        {
            size_t bestCurrentIndex = mCurvePoints.size();
            double bestDistance = std::numeric_limits<double>::max();
            for (size_t currentIndex = nextCurrentIndex; currentIndex < mCurvePoints.size(); ++currentIndex)
            {
                const double distance = cv::norm(mCurvePoints[currentIndex] - otherPoints[otherIndex]);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestCurrentIndex = currentIndex;
                }
            }
            if (bestCurrentIndex == mCurvePoints.size() || bestDistance > curveConfig->mapFusionMaximumP90Distance)
                continue;
            matches.push_back(GeometryMatch{bestCurrentIndex, otherIndex, bestDistance});
            distances.push_back(bestDistance);
            nextCurrentIndex = bestCurrentIndex + 1;
        }

        const size_t shorterCurveSize = std::min(mCurvePoints.size(), otherPoints.size());
        const double coverage = shorterCurveSize > 0 ? static_cast<double>(matches.size()) / static_cast<double>(shorterCurveSize) : 0.0;
        if (matches.size() < 2 || coverage < curveConfig->mapFusionMinimum3DCoverage)
            return false;

        std::sort(distances.begin(), distances.end());
        const double medianDistance = distances[distances.size() / 2];
        const size_t p90Index = static_cast<size_t>(std::floor(0.9 * static_cast<double>(distances.size() - 1)));
        const double p90Distance = distances[p90Index];
        if (medianDistance > curveConfig->fusionMax3DDistance || p90Distance > curveConfig->mapFusionMaximumP90Distance)
            return false;

        for (const GeometryMatch &match : matches)
        {
            const unsigned int currentWeight = std::max(1U, mCurvePointFusionWeights[match.currentIndex]);
            const unsigned int otherWeight = std::max(1U, otherWeights[match.otherIndex]);
            const double totalWeight = static_cast<double>(currentWeight) + static_cast<double>(otherWeight);
            mCurvePoints[match.currentIndex] = (mCurvePoints[match.currentIndex] * static_cast<double>(currentWeight) + otherPoints[match.otherIndex] * static_cast<double>(otherWeight)) * (1.0 / totalWeight);
            mCurvePointFusionWeights[match.currentIndex] = std::min(static_cast<unsigned int>(curveConfig->maximumFusionWeight), currentWeight + otherWeight);
        }

        const auto directionCosine = [](const cv::Point3d &first, const cv::Point3d &second)
        {
            const double denominator = cv::norm(first) * cv::norm(second);
            return denominator <= std::numeric_limits<double>::epsilon() ? -1.0 : first.dot(second) / denominator;
        };

        const GeometryMatch &firstMatch = matches.front();
        const GeometryMatch &lastMatch = matches.back();
        const size_t prefixCandidateCount = firstMatch.currentIndex == 0 ? firstMatch.otherIndex : 0;
        const size_t suffixCandidateCount = lastMatch.currentIndex + 1 == mCurvePoints.size() ? otherPoints.size() - lastMatch.otherIndex - 1 : 0;

        bool acceptPrefix = prefixCandidateCount >= static_cast<size_t>(curveConfig->minimumExtensionSamples);
        if (acceptPrefix)
        {
            const size_t lastPrefixIndex = firstMatch.otherIndex - 1;
            const double gap = cv::norm(otherPoints[lastPrefixIndex] - mCurvePoints.front());
            const double cosine = directionCosine(otherPoints[lastPrefixIndex] - otherPoints[lastPrefixIndex - 1], mCurvePoints[1] - mCurvePoints[0]);
            acceptPrefix = gap <= curveConfig->extensionMax3DGap && cosine >= curveConfig->minimumJoinDirectionCosine;
        }

        bool acceptSuffix = suffixCandidateCount >= static_cast<size_t>(curveConfig->minimumExtensionSamples);
        if (acceptSuffix)
        {
            const size_t firstSuffixIndex = lastMatch.otherIndex + 1;
            const double gap = cv::norm(otherPoints[firstSuffixIndex] - mCurvePoints.back());
            const double cosine = directionCosine(mCurvePoints.back() - mCurvePoints[mCurvePoints.size() - 2], otherPoints[firstSuffixIndex + 1] - otherPoints[firstSuffixIndex]);
            acceptSuffix = gap <= curveConfig->extensionMax3DGap && cosine >= curveConfig->minimumJoinDirectionCosine;
        }

        const size_t prefixCount = acceptPrefix ? prefixCandidateCount : 0;
        const size_t suffixCount = acceptSuffix ? suffixCandidateCount : 0;
        if (prefixCount > 0 || suffixCount > 0)
        {
            std::vector<cv::Point3d> mergedPoints;
            std::vector<unsigned int> mergedWeights;
            mergedPoints.reserve(prefixCount + mCurvePoints.size() + suffixCount);
            mergedWeights.reserve(prefixCount + mCurvePointFusionWeights.size() + suffixCount);
            if (prefixCount > 0)
            {
                mergedPoints.insert(mergedPoints.end(), otherPoints.begin(), otherPoints.begin() + firstMatch.otherIndex);
                mergedWeights.insert(mergedWeights.end(), otherWeights.begin(), otherWeights.begin() + firstMatch.otherIndex);
            }
            mergedPoints.insert(mergedPoints.end(), mCurvePoints.begin(), mCurvePoints.end());
            mergedWeights.insert(mergedWeights.end(), mCurvePointFusionWeights.begin(), mCurvePointFusionWeights.end());
            if (suffixCount > 0)
            {
                mergedPoints.insert(mergedPoints.end(), otherPoints.begin() + lastMatch.otherIndex + 1, otherPoints.end());
                mergedWeights.insert(mergedWeights.end(), otherWeights.begin() + lastMatch.otherIndex + 1, otherWeights.end());
            }
            mCurvePoints.swap(mergedPoints);
            mCurvePointFusionWeights.swap(mergedWeights);
        }
        SmoothGeometry(curveConfig);
        return true;
    }

    bool MapCurve::Replace(MapCurve *pSurvivor, const CurveConfigPtr &curveConfig)
    {
        if (!pSurvivor || pSurvivor == this || pSurvivor->isBad() || !pSurvivor->MergeGeometryFrom(this, curveConfig))
            return false;

        map<KeyFrame *, vector<size_t>> observations;
        int visibleCount = 0;
        int foundCount = 0;
        {
            unique_lock<mutex> featureLock(mMutexFeatures);
            unique_lock<mutex> positionLock(mMutexPos);
            observations = mObservations;
            mObservations.clear();
            nObs = 0;
            visibleCount = mnVisible;
            foundCount = mnFound;
            mbBad = true;
            mpReplaced = pSurvivor;
        }

        for (const auto &observation : observations)
        {
            KeyFrame *pKF = observation.first;
            for (const size_t curveIndex : observation.second)
            {
                MapCurve *pCurrent = pKF->GetMapCurve(curveIndex);
                if (pCurrent == this)
                {
                    pKF->ReplaceMapCurveMatch(curveIndex, pSurvivor);
                    pSurvivor->AddObservation(pKF, curveIndex);
                }
                else if (pCurrent == pSurvivor)
                    pSurvivor->AddObservation(pKF, curveIndex);
            }
        }

        pSurvivor->IncreaseVisible(visibleCount);
        pSurvivor->IncreaseFound(foundCount);
        mpMap->EraseMapCurve(this);

        set<KeyFrame *> affectedKeyFrames;
        for (const auto &observation : observations)
            affectedKeyFrames.insert(observation.first);
        const map<KeyFrame *, vector<size_t>> survivorObservations = pSurvivor->GetObservations();
        for (const auto &observation : survivorObservations)
            affectedKeyFrames.insert(observation.first);
        for (KeyFrame *pKF : affectedKeyFrames)
        {
            if (pKF && !pKF->isBad())
                pKF->UpdateConnections();
        }
        return true;
    }

    std::map<KeyFrame *, std::vector<size_t>> MapCurve::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapCurve::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    int MapCurve::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        const auto observation = mObservations.find(pKF);
        if (observation == mObservations.end() || observation->second.empty())
            return -1;
        return static_cast<int>(observation->second.front());
    }

    std::vector<size_t> MapCurve::GetIndicesInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        const auto observation = mObservations.find(pKF);
        return observation == mObservations.end() ? std::vector<size_t>() : observation->second;
    }

    bool ORB_SLAM2::MapCurve::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations.count(pKF);
    }

    void ORB_SLAM2::MapCurve::SetBadFlag()
    {
        map<KeyFrame *, vector<size_t>> observations;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            observations = mObservations;
            mObservations.clear();
            nObs = 0;
        }
        for (map<KeyFrame *, vector<size_t>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            for (const size_t curveIndex : mit->second)
            {
                if (pKF->GetMapCurve(curveIndex) == this)
                    pKF->EraseMapCurveMatch(curveIndex);
            }
        }

        mpMap->EraseMapCurve(this);
    }

    bool MapCurve::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mbBad;
    }
}
