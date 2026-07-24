#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include <cstddef>
#include <opencv2/core/core.hpp>
#include <vector>
#include "Curve/EdgeCluster.h"

namespace ORB_SLAM2
{
    class BezierCurve
    {
    public:
        std::vector<orderedEdgePoint> controlPoints; // 每条边缘可能有多段Bezier曲线，每段曲线由一组控制点定义
        // 按二维弧长均匀采样，采样数量由曲线长度和 spacing 决定。
        std::vector<orderedEdgePoint> sampledPoints;

        void sampleByArcLengthSpacing(int spacing, std::size_t lookupSegmentCount = 200, bool removeDuplicatePixels = true);

        void reserve(size_t controlPointCount) { controlPoints.reserve(controlPointCount); }
        void push_back(const orderedEdgePoint &point) { controlPoints.push_back(point); }
    };

    class BezierCurveFitter
    {
    public:
        explicit BezierCurveFitter(double rho_p = 1.0, std::size_t minSplitPoints = 10);

        std::vector<BezierCurve> fitAdaptive(const std::vector<orderedEdgePoint> &points) const;

    private:
        std::vector<double> chordLengthParameters(const std::vector<orderedEdgePoint> &points) const;
        std::vector<double> computeResiduals(const std::vector<orderedEdgePoint> &controlPoints,
                                             const std::vector<orderedEdgePoint> &edge,
                                             const std::vector<double> &parameters) const;
        BezierCurve fitWithEndPoints(const std::vector<orderedEdgePoint> &edge,
                                     const std::vector<double> &parameters,
                                     int order) const;

        double rho_p_;
        std::size_t minSplitPoints_;
    };
}

#endif // BEZIERCURVE_H
