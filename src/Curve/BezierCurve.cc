#include "Curve/BezierCurve.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <Eigen/Dense>
#include <limits>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM2
{
    struct ArcLengthTable
    {
        std::vector<double> lengths;

        double totalLength() const { return lengths.empty() ? 0.0 : lengths.back(); }
    };

    static inline double bernsteinBasis(int n, int i, double t)
    {
        if (i < 0 || i > n)
            return 0.0;

        t = std::max(0.0, std::min(1.0, t));
        const double u = 1.0 - t;
        double binomialCoeff = 1.0;
        for (int j = 0; j < i; ++j)
            binomialCoeff *= (n - j) / static_cast<double>(j + 1);

        return binomialCoeff * std::pow(u, n - i) * std::pow(t, i);
    }

    static inline orderedEdgePoint evaluate(const std::vector<orderedEdgePoint> &controlPoints, double t)
    {
        t = std::max(0.0, std::min(1.0, t));
        if (controlPoints.empty())
            return orderedEdgePoint(0.0, 0.0);

        if (controlPoints.size() == 1)
            return controlPoints.front();

        // 这里只需要插值二维坐标。旧实现会为每次求值复制整组
        // orderedEdgePoint，曲线拟合和弧长采样时会产生大量小对象拷贝。
        orderedEdgePoint point = controlPoints.front();
        point.x = 0.0;
        point.y = 0.0;
        const int degree = static_cast<int>(controlPoints.size()) - 1;
        for (int index = 0; index <= degree; ++index)
        {
            const double weight = bernsteinBasis(degree, index, t);
            point.x += weight * controlPoints[index].x;
            point.y += weight * controlPoints[index].y;
        }
        return point;
    }

    static inline cv::Point2d evaluateDerivative(
        const std::vector<orderedEdgePoint> &controlPoints,
        double t,
        int derivativeOrder)
    {
        const int degree = static_cast<int>(controlPoints.size()) - 1;
        if (degree < derivativeOrder || derivativeOrder < 1)
            return cv::Point2d();

        t = std::max(0.0, std::min(1.0, t));
        cv::Point2d derivative;

        if (derivativeOrder == 1)
        {
            for (int i = 0; i < degree; ++i)
            {
                const double weight = degree * bernsteinBasis(degree - 1, i, t);
                derivative.x += weight * (controlPoints[i + 1].x - controlPoints[i].x);
                derivative.y += weight * (controlPoints[i + 1].y - controlPoints[i].y);
            }
            return derivative;
        }

        if (derivativeOrder == 2)
        {
            const double scale = degree * (degree - 1.0);
            for (int i = 0; i < degree - 1; ++i)
            {
                const double weight = scale * bernsteinBasis(degree - 2, i, t);
                derivative.x += weight * (controlPoints[i + 2].x - 2.0 * controlPoints[i + 1].x + controlPoints[i].x);
                derivative.y += weight * (controlPoints[i + 2].y - 2.0 * controlPoints[i + 1].y + controlPoints[i].y);
            }
        }

        return derivative;
    }

    static inline double squaredDistance(const orderedEdgePoint &a, const orderedEdgePoint &b)
    {
        const double dx = a.x - b.x;
        const double dy = a.y - b.y;
        return dx * dx + dy * dy;
    }

    static inline bool sameRoundedPixel(const orderedEdgePoint &a, const orderedEdgePoint &b)
    {
        return cvRound(a.x) == cvRound(b.x) && cvRound(a.y) == cvRound(b.y);
    }

    static inline ArcLengthTable buildArcLengthTable(const BezierCurve &curve, std::size_t segmentCount)
    {
        segmentCount = std::max<std::size_t>(segmentCount, 1);

        ArcLengthTable table;
        table.lengths.resize(segmentCount + 1, 0.0);

        orderedEdgePoint prevPoint = evaluate(curve.controlPoints, 0.0);
        for (std::size_t i = 1; i <= segmentCount; ++i)
        {
            const double t = static_cast<double>(i) / static_cast<double>(segmentCount);
            const orderedEdgePoint point = evaluate(curve.controlPoints, t);
            table.lengths[i] = table.lengths[i - 1] + std::sqrt(squaredDistance(prevPoint, point));
            prevPoint = point;
        }

        return table;
    }

    static inline double parameterAtLength(const ArcLengthTable &table, double targetLength)
    {
        const auto upper = std::lower_bound(table.lengths.begin(), table.lengths.end(), targetLength);
        const std::size_t upperIndex = static_cast<std::size_t>(std::distance(table.lengths.begin(), upper));

        if (upperIndex == 0)
            return 0.0;
        if (upperIndex >= table.lengths.size())
            return 1.0;

        const double length0 = table.lengths[upperIndex - 1];
        const double length1 = table.lengths[upperIndex];
        const double denom = length1 - length0;
        const double segmentCount = static_cast<double>(table.lengths.size() - 1);
        if (denom <= DBL_EPSILON)
            return static_cast<double>(upperIndex - 1) / segmentCount;

        const double ratio = (targetLength - length0) / denom;
        return (static_cast<double>(upperIndex - 1) + ratio) / segmentCount;
    }

    static inline void appendSample(BezierCurve &curve, orderedEdgePoint point, bool removeDuplicatePixels)
    {
        if (removeDuplicatePixels && !curve.sampledPoints.empty() && sameRoundedPixel(curve.sampledPoints.back(), point))
            return;

        curve.sampledPoints.push_back(point);
    }

    void BezierCurve::sampleByArcLengthSpacing(int spacing, std::size_t lookupSegmentCount, bool removeDuplicatePixels)
    {
        sampledPoints.clear();
        if (controlPoints.empty())
            return;

        if (spacing <= 0)
            spacing = 1;

        lookupSegmentCount = std::max<std::size_t>(lookupSegmentCount, 1);
        ArcLengthTable table = buildArcLengthTable(*this, lookupSegmentCount);
        const double length = table.totalLength();
        // N个采样点包含N-1个弧长区间。直接按照完整曲线长度和期望
        // 像素间距确定区间数，不再对采样点总数设置上限。
        const std::size_t segmentCount = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(length / spacing)));

        if (controlPoints.size() == 1)
        {
            appendSample(*this, evaluate(controlPoints, 0.0), false);
            return;
        }

        // 通常默认的 200 段查找表已经足够，直接复用；只有实际采样段数
        // 更大时才重新构建，避免每条 Bezier 曲线重复计算两次弧长表。
        if (segmentCount > lookupSegmentCount)
            table = buildArcLengthTable(*this, segmentCount);
        const double totalLength = table.totalLength();
        sampledPoints.reserve(segmentCount + 1);

        for (std::size_t i = 0; i <= segmentCount; ++i)
        {
            const double ratio = static_cast<double>(i) / static_cast<double>(segmentCount);
            const double t = totalLength > DBL_EPSILON
                                 ? parameterAtLength(table, totalLength * ratio)
                                 : ratio;
            appendSample(*this, evaluate(controlPoints, t), removeDuplicatePixels);
        }
    }

    BezierCurveFitter::BezierCurveFitter(double rho_p, std::size_t minSplitPoints)
        : rho_p_(rho_p), minSplitPoints_(std::max<std::size_t>(2, minSplitPoints))
    {
    }

    std::vector<double> BezierCurveFitter::chordLengthParameters(const std::vector<orderedEdgePoint> &edge) const
    {
        const size_t n = edge.size();
        std::vector<double> parameters(n, 0.0);
        if (n < 2)
            return parameters;

        for (size_t i = 1; i < n; ++i)
            parameters[i] = parameters[i - 1] + std::sqrt(squaredDistance(edge[i - 1], edge[i]));

        const double totalLength = parameters.back();
        if (totalLength <= DBL_EPSILON)
        {
            for (size_t i = 0; i < n; ++i)
                parameters[i] = static_cast<double>(i) / static_cast<double>(n - 1);
            return parameters;
        }

        for (double &parameter : parameters)
            parameter /= totalLength;

        return parameters;
    }

    std::vector<double> BezierCurveFitter::computeResiduals(
        const std::vector<orderedEdgePoint> &controlPoints,
        const std::vector<orderedEdgePoint> &edge,
        const std::vector<double> &parameters) const
    {
        const size_t n = edge.size();
        std::vector<double> residuals;
        residuals.reserve(n);

        for (size_t i = 0; i < n; ++i)
        {
            const orderedEdgePoint &edgePoint = edge[i];
            const orderedEdgePoint curvePoint = evaluate(controlPoints, parameters[i]);
            residuals.push_back(std::sqrt(squaredDistance(curvePoint, edgePoint)));
        }

        return residuals;
    }

    std::vector<BezierCurve> BezierCurveFitter::fitAdaptive(const std::vector<orderedEdgePoint> &edge) const
    {
        std::vector<BezierCurve> fittedCurves;
        std::vector<std::vector<orderedEdgePoint>> pendingSegments;
        pendingSegments.push_back(edge);

        while (!pendingSegments.empty())
        {
            std::vector<orderedEdgePoint> segment = std::move(pendingSegments.back());
            pendingSegments.pop_back();

            if (segment.size() < 10)
                continue;

            std::vector<double> residuals;
            BezierCurve curve;
            bool fitted = false;
            // 同一分段尝试不同阶数时弦长参数不变，只计算一次。
            const std::vector<double> parameters = chordLengthParameters(segment);

            for (int order = 1; order <= 3; ++order)
            {
                curve = fitWithEndPoints(segment, parameters, order);
                if (curve.controlPoints.empty())
                {
                    fitted = true;
                    break;
                }

                residuals = computeResiduals(curve.controlPoints, segment, parameters);
                double maxResidual = *std::max_element(residuals.begin(), residuals.end());

                if (maxResidual <= rho_p_)
                {
                    fittedCurves.push_back(curve);
                    fitted = true;
                    break;
                }
            }

            if (fitted)
                continue;

            if (residuals.empty() || curve.controlPoints.empty())
                continue;

            if (segment.size() < 2 * minSplitPoints_ - 1)
            {
                fittedCurves.push_back(curve);
                continue;
            }

            const size_t firstValidSplit = minSplitPoints_ - 1;
            const size_t lastValidSplit = segment.size() - minSplitPoints_;
            auto maxIt = std::max_element(residuals.begin() + firstValidSplit,
                                          residuals.begin() + lastValidSplit + 1);
            size_t splitIndex = static_cast<size_t>(std::distance(residuals.begin(), maxIt));

            if (splitIndex < firstValidSplit || splitIndex > lastValidSplit)
            {
                fittedCurves.push_back(curve);
                continue;
            }

            std::vector<orderedEdgePoint> points1(segment.begin(), segment.begin() + splitIndex + 1);
            std::vector<orderedEdgePoint> points2(segment.begin() + splitIndex, segment.end());

            pendingSegments.push_back(std::move(points2));
            pendingSegments.push_back(std::move(points1));
        }

        return fittedCurves;
    }

    BezierCurve BezierCurveFitter::fitWithEndPoints(
        const std::vector<orderedEdgePoint> &edge,
        const std::vector<double> &parameters,
        int order) const
    {
        int n = edge.size();

        BezierCurve curve;
        orderedEdgePoint startPoint = edge[0];
        orderedEdgePoint endPoint = edge[n - 1];

        if (order == 1)
        {
            curve.push_back(startPoint);
            curve.push_back(endPoint);
            return curve;
        }

        int unknowns = order - 1;
        Eigen::MatrixXd A(n, unknowns);
        Eigen::VectorXd b_x(n);
        Eigen::VectorXd b_y(n);
        for (int i = 0; i < n; ++i)
        {
            double t = parameters[i];

            double B_0 = bernsteinBasis(order, 0, t);
            double B_end = bernsteinBasis(order, order, t);

            double known_x = B_0 * startPoint.x + B_end * endPoint.x;
            double known_y = B_0 * startPoint.y + B_end * endPoint.y;

            for (int k = 1; k <= unknowns; ++k)
                A(i, k - 1) = bernsteinBasis(order, k, t);

            const orderedEdgePoint &q_i = edge[i];
            b_x(i) = q_i.x - known_x;
            b_y(i) = q_i.y - known_y;
        }

        // A 对 x/y 相同，QR 分解只需执行一次。
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomposition(A);
        Eigen::VectorXd middle_x = decomposition.solve(b_x);
        Eigen::VectorXd middle_y = decomposition.solve(b_y);

        curve.reserve(order + 1);
        curve.push_back(startPoint);

        for (int k = 0; k < unknowns; ++k)
        {
            curve.push_back(orderedEdgePoint(middle_x(k), middle_y(k)));
        }

        curve.push_back(endPoint);
        return curve;
    }
}
