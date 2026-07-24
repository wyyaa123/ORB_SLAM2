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
        // A frame-local identifier of the ordered Edge that produced this
        // Bezier segment. All segments fitted from the same Edge share it.
        std::size_t edgeChainId = -1;
        // Position of this segment in the original ordered Edge and the total
        // number of fitted segments before depth filtering.
        std::size_t segmentIndex = 0;
        std::size_t segmentCount = 1;

        std::vector<orderedEdgePoint> controlPoints; // 每条边缘可能有多段Bezier曲线，每段曲线由一组控制点定义
        // 按二维弧长均匀采样，采样数量由曲线长度和 spacing 决定。
        std::vector<orderedEdgePoint> sampledPoints;

        bool hasEdgeChain() const { return edgeChainId != -1; }

        void sampleByArcLengthSpacing(int spacing, std::size_t lookupSegmentCount = 200, bool removeDuplicatePixels = true);

        void reserve(size_t controlPointCount) { controlPoints.reserve(controlPointCount); }
        void push_back(const orderedEdgePoint &point) { controlPoints.push_back(point); }
    };

    class BezierCurveFitter
    {
    public:
        explicit BezierCurveFitter(double rho_p = 1.0, std::size_t minSplitPoints = 10);

        std::vector<BezierCurve> fitAdaptive(const std::vector<orderedEdgePoint> &points, std::size_t edgeChainId = -1) const;

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
