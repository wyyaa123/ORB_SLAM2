#include "Curve/CurveGeometry.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ORB_SLAM2
{
    namespace
    {
        const double kPi = 3.14159265358979323846;
        const double kMinimumVectorNorm = 1e-9;
        const double kMinimumTrackedTurnRadians = 1e-4;
        const double kCollinearDirectionCosine = 1.0 - 1e-10;
        const double kMaximumSharedEndpointPixelDistance = 1.5;

        bool IsFinitePoint(const cv::Point3d &point)
        {
            return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
        }

        bool IsValidDepthSample(const orderedEdgePoint &point, const CurveConfig &curveConfig)
        {
            return std::isfinite(point.depth) &&
                   point.depth > curveConfig.minDepth &&
                   point.depth < curveConfig.maxDepth &&
                   std::isfinite(point.x_3d) &&
                   std::isfinite(point.y_3d) &&
                   std::isfinite(point.z_3d);
        }

        cv::Point3d CameraPoint(const orderedEdgePoint &point)
        {
            return cv::Point3d(point.x_3d, point.y_3d, point.z_3d);
        }

        double ClampUnit(const double value)
        {
            return std::max(-1.0, std::min(1.0, value));
        }

        cv::Point3d CrossProduct(
            const cv::Point3d &first,
            const cv::Point3d &second)
        {
            return cv::Point3d(
                first.y * second.z - first.z * second.y,
                first.z * second.x - first.x * second.z,
                first.x * second.y - first.y * second.x);
        }

        std::vector<cv::Point3d> BuildGeometryVertices(
            const std::vector<cv::Point3d> &points)
        {
            std::vector<cv::Point3d> geometryVertices;
            if (points.empty())
                return geometryVertices;

            geometryVertices.reserve(points.size());
            for (const cv::Point3d &point : points)
            {
                if (!geometryVertices.empty() &&
                    cv::norm(point - geometryVertices.back()) <=
                        kMinimumVectorNorm)
                {
                    continue;
                }

                while (geometryVertices.size() >= 2)
                {
                    const cv::Point3d incoming =
                        geometryVertices.back() -
                        geometryVertices[geometryVertices.size() - 2];
                    const cv::Point3d outgoing =
                        point - geometryVertices.back();
                    const double denominator =
                        cv::norm(incoming) * cv::norm(outgoing);
                    if (denominator <= kMinimumVectorNorm ||
                        incoming.dot(outgoing) / denominator <
                            kCollinearDirectionCosine)
                    {
                        break;
                    }
                    geometryVertices.pop_back();
                }
                geometryVertices.push_back(point);
            }

            return geometryVertices;
        }
    } // namespace

    CurveGeometryQuality::CurveGeometryQuality()
        : allFinite(true),
          pointCount(0),
          evaluationPointCount(0),
          arcLength(0.0),
          gapCount(0),
          validTurnCount(0),
          bendEventCount(0),
          highTurnCount(0),
          hardUTurnCount(0),
          bendPairCount(0),
          bendFlipCount(0),
          locallyOscillating(false),
          gapRatio(0.0),
          highTurnRatio(0.0),
          bendFlipRatio(0.0),
          p90TurnDegrees(0.0)
    {
    }

    std::vector<ContinuousCurveSegment> SplitContinuousCurveSamples(
        const std::vector<orderedEdgePoint> &samples,
        const CurveConfig &curveConfig)
    {
        std::vector<ContinuousCurveSegment> segments;
        ContinuousCurveSegment currentSegment;
        const std::size_t minimumSamples =
            static_cast<std::size_t>(std::max(
                4, curveConfig.minimumDepthSegmentSamples));

        const auto flushSegment = [&]()
        {
            if (currentSegment.samples.size() >= minimumSamples)
                segments.push_back(currentSegment);
            currentSegment = ContinuousCurveSegment();
        };

        for (std::size_t sampleIndex = 0;
             sampleIndex < samples.size();
             ++sampleIndex)
        {
            const orderedEdgePoint &sample = samples[sampleIndex];
            if (!IsValidDepthSample(sample, curveConfig))
            {
                flushSegment();
                continue;
            }

            if (!currentSegment.samples.empty())
            {
                const double gap =
                    cv::norm(CameraPoint(sample) -
                             CameraPoint(currentSegment.samples.back()));
                if (!std::isfinite(gap) || gap > curveConfig.depthSegmentMax3DGap)
                    flushSegment();
            }

            if (currentSegment.samples.empty())
            {
                currentSegment.sourceBeginIndex = sampleIndex;
                currentSegment.sourceSampleCount = samples.size();
            }
            currentSegment.samples.push_back(sample);
            currentSegment.sourceEndIndex = sampleIndex;
        }
        flushSegment();
        return segments;
    }

    bool CanShareCurveContinuityGroup(
        const ContinuousCurveSegment &previousDepthSegment,
        const std::size_t previousBezierSegmentIndex,
        const ContinuousCurveSegment &currentDepthSegment,
        const std::size_t currentBezierSegmentIndex,
        const CurveConfig &curveConfig)
    {
        if (previousDepthSegment.samples.empty() ||
            currentDepthSegment.samples.empty() ||
            currentBezierSegmentIndex != previousBezierSegmentIndex + 1 ||
            !previousDepthSegment.endsAtSourceEnd() ||
            !currentDepthSegment.beginsAtSourceStart())
        {
            return false;
        }

        const orderedEdgePoint &previousSample =
            previousDepthSegment.samples.back();
        const orderedEdgePoint &currentSample =
            currentDepthSegment.samples.front();
        if (!IsValidDepthSample(previousSample, curveConfig) ||
            !IsValidDepthSample(currentSample, curveConfig))
        {
            return false;
        }

        const double pixelDeltaX = previousSample.x - currentSample.x;
        const double pixelDeltaY = previousSample.y - currentSample.y;
        const double pixelDistance =
            std::sqrt(pixelDeltaX * pixelDeltaX + pixelDeltaY * pixelDeltaY);
        const double spatialDistance =
            cv::norm(CameraPoint(previousSample) - CameraPoint(currentSample));
        return std::isfinite(pixelDistance) &&
               pixelDistance <= kMaximumSharedEndpointPixelDistance &&
               std::isfinite(spatialDistance) &&
               spatialDistance <= curveConfig.depthSegmentMax3DGap;
    }

    CurveGeometryQuality EvaluateCurveGeometry(
        const std::vector<cv::Point3d> &points,
        const CurveConfig &curveConfig)
    {
        CurveGeometryQuality quality;
        quality.pointCount = points.size();
        if (points.empty())
            return quality;

        for (std::size_t index = 0; index < points.size(); ++index)
        {
            if (!IsFinitePoint(points[index]))
                quality.allFinite = false;
            if (index == 0)
                continue;

            const double gap = cv::norm(points[index] - points[index - 1]);
            if (std::isfinite(gap))
                quality.arcLength += gap;
            const bool continuous = IsFinitePoint(points[index - 1]) &&
                                    IsFinitePoint(points[index]) &&
                                    std::isfinite(gap) &&
                                    gap <= curveConfig.extensionMax3DGap;
            if (!continuous)
                ++quality.gapCount;
        }

        if (points.size() > 1)
            quality.gapRatio = static_cast<double>(quality.gapCount) /
                               static_cast<double>(points.size() - 1);

        // A discontinuous polyline is rejected by the gap gate below. Do not
        // invent a straight interpolation across that discontinuity while
        // computing its curvature diagnostics.
        if (!quality.allFinite || quality.gapCount > 0)
            return quality;

        const std::vector<cv::Point3d> evaluationPoints =
            BuildGeometryVertices(points);
        quality.evaluationPointCount = evaluationPoints.size();
        if (evaluationPoints.size() < 3)
            return quality;

        std::vector<double> evaluationArcLengths(
            evaluationPoints.size(), 0.0);
        for (std::size_t pointIndex = 1;
             pointIndex < evaluationPoints.size();
             ++pointIndex)
        {
            evaluationArcLengths[pointIndex] =
                evaluationArcLengths[pointIndex - 1] +
                cv::norm(
                    evaluationPoints[pointIndex] -
                    evaluationPoints[pointIndex - 1]);
        }

        const double highTurnRadians =
            static_cast<double>(curveConfig.geometryQualityHighTurnDegrees) * kPi / 180.0;
        const double minimumBendRadians =
            static_cast<double>(curveConfig.geometryQualityMinimumBendDegrees) *
            kPi / 180.0;
        const double hardUTurnRadians =
            static_cast<double>(curveConfig.geometryQualityHardUTurnDegrees) *
            kPi / 180.0;
        std::vector<double> turnAngles;
        turnAngles.reserve(evaluationPoints.size() - 2);

        bool eventIsActive = false;
        double eventTurnAngle = 0.0;
        double eventWeightedArcLength = 0.0;
        double lastTurnArcLength = 0.0;
        cv::Point3d lastTurnAxis;
        cv::Point3d eventTurnAxisSum;
        std::vector<cv::Point3d> eventTurnAxes;
        std::vector<double> eventArcLengths;

        const auto finishEvent = [&]()
        {
            if (!eventIsActive)
                return;
            if (eventTurnAngle < minimumBendRadians)
            {
                eventIsActive = false;
                return;
            }

            turnAngles.push_back(eventTurnAngle);
            const double axisSumNorm = cv::norm(eventTurnAxisSum);
            eventTurnAxes.push_back(
                axisSumNorm > kMinimumVectorNorm
                    ? eventTurnAxisSum * (1.0 / axisSumNorm)
                    : lastTurnAxis);
            eventArcLengths.push_back(
                eventWeightedArcLength / eventTurnAngle);
            eventIsActive = false;
        };

        for (std::size_t index = 1;
             index + 1 < evaluationPoints.size();
             ++index)
        {
            cv::Point3d incoming =
                evaluationPoints[index] - evaluationPoints[index - 1];
            cv::Point3d outgoing =
                evaluationPoints[index + 1] - evaluationPoints[index];
            const double incomingNorm = cv::norm(incoming);
            const double outgoingNorm = cv::norm(outgoing);
            if (incomingNorm <= kMinimumVectorNorm || outgoingNorm <= kMinimumVectorNorm)
            {
                finishEvent();
                continue;
            }

            incoming *= 1.0 / incomingNorm;
            outgoing *= 1.0 / outgoingNorm;
            ++quality.validTurnCount;
            const double turnAngle = std::acos(ClampUnit(incoming.dot(outgoing)));
            if (turnAngle >= hardUTurnRadians)
            {
                finishEvent();
                ++quality.hardUTurnCount;
                continue;
            }

            cv::Point3d turnAxis = CrossProduct(incoming, outgoing);
            const double turnAxisNorm = cv::norm(turnAxis);
            if (turnAngle < kMinimumTrackedTurnRadians ||
                turnAxisNorm <= kMinimumVectorNorm)
            {
                finishEvent();
                continue;
            }

            turnAxis *= 1.0 / turnAxisNorm;
            const bool continuesSameEvent =
                eventIsActive &&
                evaluationArcLengths[index] - lastTurnArcLength <=
                    curveConfig.geometryQualityMaximumBendSeparation &&
                lastTurnAxis.dot(turnAxis) >= 0.0;
            if (!continuesSameEvent)
            {
                finishEvent();
                eventIsActive = true;
                eventTurnAngle = 0.0;
                eventWeightedArcLength = 0.0;
                eventTurnAxisSum = cv::Point3d();
            }

            eventTurnAngle += turnAngle;
            eventWeightedArcLength +=
                evaluationArcLengths[index] * turnAngle;
            eventTurnAxisSum += turnAxis * turnAngle;
            lastTurnAxis = turnAxis;
            lastTurnArcLength = evaluationArcLengths[index];
        }
        finishEvent();

        quality.bendEventCount = turnAngles.size();
        std::vector<std::size_t> highTurnEventIndices;
        highTurnEventIndices.reserve(turnAngles.size());
        for (std::size_t eventIndex = 0;
             eventIndex < turnAngles.size();
             ++eventIndex)
        {
            if (turnAngles[eventIndex] > highTurnRadians)
            {
                ++quality.highTurnCount;
                highTurnEventIndices.push_back(eventIndex);
            }
        }

        for (std::size_t highEventIndex = 1;
             highEventIndex < highTurnEventIndices.size();
             ++highEventIndex)
        {
            const std::size_t previousEventIndex =
                highTurnEventIndices[highEventIndex - 1];
            const std::size_t currentEventIndex =
                highTurnEventIndices[highEventIndex];
            if (eventArcLengths[currentEventIndex] -
                    eventArcLengths[previousEventIndex] >
                curveConfig.geometryQualityMaximumBendSeparation)
            {
                continue;
            }
            ++quality.bendPairCount;
            if (eventTurnAxes[previousEventIndex].dot(
                    eventTurnAxes[currentEventIndex]) <
                curveConfig.geometryQualityBendFlipCosine)
            {
                ++quality.bendFlipCount;
            }
        }

        // Search fixed physical windows over high-turn events. Straight tails
        // and arbitrarily long chains of low-angle bends are deliberately not
        // allowed to dilute a short, strongly oscillating section.
        for (std::size_t windowBegin = 0;
             windowBegin < highTurnEventIndices.size() &&
             !quality.locallyOscillating;
             ++windowBegin)
        {
            std::size_t localFlipCount = 0;
            for (std::size_t windowEnd = windowBegin;
                 windowEnd < highTurnEventIndices.size();
                 ++windowEnd)
            {
                const std::size_t firstEventIndex =
                    highTurnEventIndices[windowBegin];
                const std::size_t lastEventIndex =
                    highTurnEventIndices[windowEnd];
                const double eventSpan =
                    eventArcLengths[lastEventIndex] -
                    eventArcLengths[firstEventIndex];
                if (eventSpan >
                    curveConfig.geometryQualityMaximumBendSeparation)
                {
                    break;
                }

                if (windowEnd > windowBegin)
                {
                    const std::size_t previousEventIndex =
                        highTurnEventIndices[windowEnd - 1];
                    if (eventTurnAxes[previousEventIndex].dot(
                            eventTurnAxes[lastEventIndex]) <
                        curveConfig.geometryQualityBendFlipCosine)
                    {
                        ++localFlipCount;
                    }
                }

                const std::size_t localHighTurnCount =
                    windowEnd - windowBegin + 1;
                const std::size_t localPairCount =
                    localHighTurnCount - 1;
                const double windowSpan =
                    eventSpan +
                    curveConfig.geometryQualityReferenceSpacing;
                const double physicalTurnSlotCount = std::max(
                    1.0,
                    std::ceil(
                        windowSpan /
                        curveConfig.geometryQualityReferenceSpacing));
                const double ratioDenominator = std::max(
                    physicalTurnSlotCount,
                    static_cast<double>(localHighTurnCount));
                const double localHighTurnRatio =
                    static_cast<double>(localHighTurnCount) /
                    ratioDenominator;
                const double localBendFlipRatio =
                    localPairCount > 0
                        ? static_cast<double>(localFlipCount) /
                              static_cast<double>(localPairCount)
                        : 0.0;
                if (localHighTurnCount >=
                    static_cast<std::size_t>(
                        curveConfig.geometryQualityMinimumHighTurns))
                {
                    quality.highTurnRatio = std::max(
                        quality.highTurnRatio, localHighTurnRatio);
                    quality.bendFlipRatio = std::max(
                        quality.bendFlipRatio, localBendFlipRatio);
                }

                if (localHighTurnCount >=
                        static_cast<std::size_t>(
                            curveConfig.geometryQualityMinimumHighTurns) &&
                    localFlipCount >=
                        static_cast<std::size_t>(
                            curveConfig.geometryQualityMinimumBendFlips) &&
                    localHighTurnRatio >
                        curveConfig.geometryQualityMaximumHighTurnRatio &&
                    localBendFlipRatio >
                        curveConfig.geometryQualityMaximumBendFlipRatio)
                {
                    quality.locallyOscillating = true;
                    break;
                }
            }
        }

        if (!turnAngles.empty())
        {
            std::sort(turnAngles.begin(), turnAngles.end());
            const std::size_t p90Index = static_cast<std::size_t>(
                std::floor(0.9 * static_cast<double>(turnAngles.size() - 1)));
            quality.p90TurnDegrees = turnAngles[p90Index] * 180.0 / kPi;
        }
        return quality;
    }

    bool IsCurveGeometryAcceptable(
        const std::vector<cv::Point3d> &points,
        const CurveConfig &curveConfig)
    {
        if (points.size() <
            static_cast<std::size_t>(std::max(
                4, curveConfig.minimumDepthSegmentSamples)))
            return false;

        const CurveGeometryQuality quality = EvaluateCurveGeometry(points, curveConfig);
        if (!quality.allFinite ||
            quality.gapCount > 0 ||
            quality.arcLength <= kMinimumVectorNorm ||
            quality.evaluationPointCount < 2)
        {
            return false;
        }

        const bool repeatedlyFoldingBack =
            quality.hardUTurnCount >=
            static_cast<std::size_t>(
                curveConfig.geometryQualityMinimumHardUTurns);
        return !quality.locallyOscillating && !repeatedlyFoldingBack;
    }

    void SmoothCurveGeometry(
        std::vector<cv::Point3d> &points,
        const CurveConfig &curveConfig)
    {
        if (curveConfig.geometrySmoothingWeight <= 0.0f || points.size() < 3)
            return;

        std::vector<cv::Point3d> smoothedPoints = points;
        for (std::size_t pointIndex = 1; pointIndex + 1 < points.size(); ++pointIndex)
        {
            const cv::Point3d &previousPoint = points[pointIndex - 1];
            const cv::Point3d &currentPoint = points[pointIndex];
            const cv::Point3d &nextPoint = points[pointIndex + 1];
            if (!IsFinitePoint(previousPoint) ||
                !IsFinitePoint(currentPoint) ||
                !IsFinitePoint(nextPoint) ||
                cv::norm(currentPoint - previousPoint) > curveConfig.extensionMax3DGap ||
                cv::norm(nextPoint - currentPoint) > curveConfig.extensionMax3DGap)
            {
                continue;
            }

            const cv::Point3d localMidpoint = (previousPoint + nextPoint) * 0.5;
            const cv::Point3d correction = localMidpoint - currentPoint;
            if (cv::norm(correction) > curveConfig.fusionMax3DDistance)
                continue;

            smoothedPoints[pointIndex] =
                currentPoint + correction * curveConfig.geometrySmoothingWeight;
        }
        points.swap(smoothedPoints);
    }

} // namespace ORB_SLAM2
