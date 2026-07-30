#ifndef CURVEGEOMETRY_H
#define CURVEGEOMETRY_H

#include "Curve/CurveConfig.h"
#include "Curve/EdgeCluster.h"

#include <cstddef>
#include <opencv2/core/core.hpp>
#include <vector>

namespace ORB_SLAM2
{

    struct CurveGeometryQuality
    {
        CurveGeometryQuality();

        bool allFinite;
        std::size_t pointCount;
        std::size_t evaluationPointCount;
        double arcLength;
        std::size_t gapCount;
        std::size_t validTurnCount;
        std::size_t bendEventCount;
        std::size_t highTurnCount;
        std::size_t hardUTurnCount;
        std::size_t bendPairCount;
        std::size_t bendFlipCount;
        bool locallyOscillating;
        double gapRatio;
        double highTurnRatio;
        double bendFlipRatio;
        double p90TurnDegrees;
    };

    struct ContinuousCurveSegment
    {
        std::vector<orderedEdgePoint> samples;
        std::size_t sourceBeginIndex = 0;
        std::size_t sourceEndIndex = 0;
        std::size_t sourceSampleCount = 0;

        bool beginsAtSourceStart() const
        {
            return sourceSampleCount > 0 && sourceBeginIndex == 0;
        }

        bool endsAtSourceEnd() const
        {
            return sourceSampleCount > 0 &&
                   sourceEndIndex + 1 == sourceSampleCount;
        }
    };

    std::vector<ContinuousCurveSegment> SplitContinuousCurveSamples(
        const std::vector<orderedEdgePoint> &samples,
        const CurveConfig &curveConfig);

    // Adaptive Bezier segments overlap at their common 2-D endpoint. Only
    // fully valid neighboring segments may share a continuity group; fragments
    // split by invalid depth or a large 3-D jump must remain separate.
    bool CanShareCurveContinuityGroup(
        const ContinuousCurveSegment &previousDepthSegment,
        std::size_t previousBezierSegmentIndex,
        const ContinuousCurveSegment &currentDepthSegment,
        std::size_t currentBezierSegmentIndex,
        const CurveConfig &curveConfig);

    CurveGeometryQuality EvaluateCurveGeometry(
        const std::vector<cv::Point3d> &points,
        const CurveConfig &curveConfig);

    bool IsCurveGeometryAcceptable(
        const std::vector<cv::Point3d> &points,
        const CurveConfig &curveConfig);

    void SmoothCurveGeometry(
        std::vector<cv::Point3d> &points,
        const CurveConfig &curveConfig);

} // namespace ORB_SLAM2

#endif // CURVEGEOMETRY_H
