#include "Curve/BezierCurve.h"
#include "Curve/CurveConfig.h"
#include "Curve/CurveGeometry.h"
#include "Curve/MapCurve.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{
    int gFailureCount = 0;

#define CHECK_TRUE(condition)                                                                  \
    do                                                                                         \
    {                                                                                          \
        if (!(condition))                                                                      \
        {                                                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " #condition << '\n'; \
            ++gFailureCount;                                                                   \
        }                                                                                      \
    } while (false)

    ORB_SLAM2::orderedEdgePoint MakeDepthSample(
        const double x,
        const double y,
        const float depth)
    {
        ORB_SLAM2::orderedEdgePoint point(x, y);
        point.depth = depth;
        if (std::isfinite(depth) && depth > 0.0f)
        {
            point.x_3d = x;
            point.y_3d = y;
            point.z_3d = depth;
        }
        return point;
    }

    std::vector<cv::Point3d> MakeLine(const std::size_t count)
    {
        std::vector<cv::Point3d> points;
        points.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
            points.push_back(cv::Point3d(0.02 * index, 0.0, 1.0));
        return points;
    }

    std::vector<cv::Point3d> MakeArc(const std::size_t count)
    {
        const double halfPi = 1.57079632679489661923;
        std::vector<cv::Point3d> points;
        points.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double angle = halfPi * static_cast<double>(index) /
                                 static_cast<double>(count - 1);
            points.push_back(cv::Point3d(std::cos(angle), std::sin(angle), 1.0));
        }
        return points;
    }

    std::vector<cv::Point3d> MakeSingleCorner()
    {
        return {
            cv::Point3d(0.00, 0.00, 1.0),
            cv::Point3d(0.02, 0.00, 1.0),
            cv::Point3d(0.04, 0.00, 1.0),
            cv::Point3d(0.06, 0.00, 1.0),
            cv::Point3d(0.08, 0.00, 1.0),
            cv::Point3d(0.08, 0.02, 1.0),
            cv::Point3d(0.08, 0.04, 1.0),
            cv::Point3d(0.08, 0.06, 1.0),
            cv::Point3d(0.08, 0.08, 1.0)};
    }

    std::vector<cv::Point3d> MakeCircle(
        const double radius,
        const std::size_t count)
    {
        const double twoPi = 6.28318530717958647692;
        std::vector<cv::Point3d> points;
        points.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double angle =
                twoPi * static_cast<double>(index) /
                static_cast<double>(count - 1);
            points.push_back(cv::Point3d(
                radius * std::cos(angle),
                radius * std::sin(angle),
                1.0));
        }
        return points;
    }

    std::vector<cv::Point3d> MakeZigzag(
        const std::size_t count,
        const double xStep = 0.03,
        const double amplitude = 0.025)
    {
        std::vector<cv::Point3d> points;
        points.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double y =
                index % 2 == 0 ? amplitude : -amplitude;
            points.push_back(
                cv::Point3d(xStep * index, y, 1.0));
        }
        return points;
    }

    std::vector<cv::Point3d> MakeSameDirectionTriangle()
    {
        const double height = 0.10392304845413264;
        return {
            cv::Point3d(0.00, 0.00, 1.0),
            cv::Point3d(0.06, 0.00, 1.0),
            cv::Point3d(0.00, height, 1.0),
            cv::Point3d(-0.06, 0.00, 1.0),
            cv::Point3d(0.00, 0.00, 1.0)};
    }

    std::vector<cv::Point3d> MakeSparseAlternatingCorners()
    {
        return {
            cv::Point3d(0.0, 0.0, 1.0),
            cv::Point3d(0.1, 0.0, 1.0),
            cv::Point3d(0.2, 0.0, 1.0),
            cv::Point3d(0.3, 0.0, 1.0),
            cv::Point3d(0.3, 0.1, 1.0),
            cv::Point3d(0.3, 0.2, 1.0),
            cv::Point3d(0.3, 0.3, 1.0),
            cv::Point3d(0.4, 0.3, 1.0),
            cv::Point3d(0.5, 0.3, 1.0),
            cv::Point3d(0.6, 0.3, 1.0),
            cv::Point3d(0.6, 0.4, 1.0),
            cv::Point3d(0.6, 0.5, 1.0),
            cv::Point3d(0.6, 0.6, 1.0)};
    }

    std::vector<cv::Point3d> MakeSineCurve(
        const std::size_t periodCount,
        const std::size_t samplesPerPeriod)
    {
        const double twoPi = 6.28318530717958647692;
        const double wavelength = 0.04;
        const double amplitude = 0.008;
        const std::size_t intervalCount =
            periodCount * samplesPerPeriod;
        std::vector<cv::Point3d> points;
        points.reserve(intervalCount + 1);
        for (std::size_t sampleIndex = 0;
             sampleIndex <= intervalCount;
             ++sampleIndex)
        {
            const double phase =
                twoPi * static_cast<double>(sampleIndex) /
                static_cast<double>(samplesPerPeriod);
            points.push_back(cv::Point3d(
                wavelength * static_cast<double>(sampleIndex) /
                    static_cast<double>(samplesPerPeriod),
                amplitude * std::sin(phase),
                1.0));
        }
        return points;
    }

    std::vector<cv::Point3d> DensifyPolyline(
        const std::vector<cv::Point3d> &points,
        const std::size_t factor)
    {
        if (points.size() < 2 || factor <= 1)
            return points;

        std::vector<cv::Point3d> densePoints;
        densePoints.reserve((points.size() - 1) * factor + 1);
        for (std::size_t pointIndex = 1;
             pointIndex < points.size();
             ++pointIndex)
        {
            for (std::size_t step = 0; step < factor; ++step)
            {
                const double ratio =
                    static_cast<double>(step) /
                    static_cast<double>(factor);
                densePoints.push_back(
                    points[pointIndex - 1] * (1.0 - ratio) +
                    points[pointIndex] * ratio);
            }
        }
        densePoints.push_back(points.back());
        return densePoints;
    }

    void CheckEquivalentQuality(
        const ORB_SLAM2::CurveGeometryQuality &expected,
        const ORB_SLAM2::CurveGeometryQuality &actual)
    {
        CHECK_TRUE(actual.allFinite == expected.allFinite);
        CHECK_TRUE(actual.evaluationPointCount == expected.evaluationPointCount);
        CHECK_TRUE(std::abs(actual.arcLength - expected.arcLength) < 1e-9);
        CHECK_TRUE(actual.gapCount == expected.gapCount);
        CHECK_TRUE(actual.validTurnCount == expected.validTurnCount);
        CHECK_TRUE(actual.bendEventCount == expected.bendEventCount);
        CHECK_TRUE(actual.highTurnCount == expected.highTurnCount);
        CHECK_TRUE(actual.hardUTurnCount == expected.hardUTurnCount);
        CHECK_TRUE(actual.bendPairCount == expected.bendPairCount);
        CHECK_TRUE(actual.bendFlipCount == expected.bendFlipCount);
        CHECK_TRUE(actual.locallyOscillating == expected.locallyOscillating);
        CHECK_TRUE(std::abs(actual.highTurnRatio - expected.highTurnRatio) < 1e-9);
        CHECK_TRUE(std::abs(actual.bendFlipRatio - expected.bendFlipRatio) < 1e-9);
        CHECK_TRUE(std::abs(actual.p90TurnDegrees - expected.p90TurnDegrees) < 1e-7);
    }

    void TestDepthSegmentation(const ORB_SLAM2::CurveConfig &config)
    {
        std::vector<ORB_SLAM2::orderedEdgePoint> invalidGap = {
            MakeDepthSample(0.00, 0.0, 1.0f),
            MakeDepthSample(0.01, 0.0, 1.0f),
            MakeDepthSample(0.02, 0.0, 1.0f),
            MakeDepthSample(0.03, 0.0, 1.0f),
            MakeDepthSample(0.04, 0.0, 0.0f),
            MakeDepthSample(0.05, 0.0, 0.0f),
            MakeDepthSample(0.06, 0.0, 1.0f),
            MakeDepthSample(0.07, 0.0, 1.0f),
            MakeDepthSample(0.08, 0.0, 1.0f),
            MakeDepthSample(0.09, 0.0, 1.0f)};
        const auto invalidSegments =
            ORB_SLAM2::SplitContinuousCurveSamples(invalidGap, config);
        CHECK_TRUE(invalidSegments.size() == 2);
        CHECK_TRUE(invalidSegments[0].samples.size() == 4);
        CHECK_TRUE(invalidSegments[1].samples.size() == 4);
        CHECK_TRUE(invalidSegments[0].sourceBeginIndex == 0);
        CHECK_TRUE(invalidSegments[0].sourceEndIndex == 3);
        CHECK_TRUE(invalidSegments[1].sourceBeginIndex == 6);
        CHECK_TRUE(invalidSegments[1].sourceEndIndex == 9);
        CHECK_TRUE(
            cv::norm(
                cv::Point3d(
                    invalidSegments[0].samples.back().x_3d,
                    invalidSegments[0].samples.back().y_3d,
                    invalidSegments[0].samples.back().z_3d) -
                cv::Point3d(
                    invalidSegments[1].samples.front().x_3d,
                    invalidSegments[1].samples.front().y_3d,
                    invalidSegments[1].samples.front().z_3d)) <
            config.depthSegmentMax3DGap);
        CHECK_TRUE(!ORB_SLAM2::CanShareCurveContinuityGroup(
            invalidSegments[0], 0, invalidSegments[1], 0, config));

        std::vector<ORB_SLAM2::orderedEdgePoint> spatialGap = {
            MakeDepthSample(0.00, 0.0, 1.0f),
            MakeDepthSample(0.01, 0.0, 1.0f),
            MakeDepthSample(0.02, 0.0, 1.0f),
            MakeDepthSample(0.03, 0.0, 1.0f),
            MakeDepthSample(0.30, 0.0, 1.0f),
            MakeDepthSample(0.31, 0.0, 1.0f),
            MakeDepthSample(0.32, 0.0, 1.0f),
            MakeDepthSample(0.33, 0.0, 1.0f)};
        const auto spatialSegments =
            ORB_SLAM2::SplitContinuousCurveSamples(spatialGap, config);
        CHECK_TRUE(spatialSegments.size() == 2);
        CHECK_TRUE(spatialSegments[0].samples.size() == 4);
        CHECK_TRUE(spatialSegments[1].samples.size() == 4);
        CHECK_TRUE(!ORB_SLAM2::CanShareCurveContinuityGroup(
            spatialSegments[0], 0, spatialSegments[1], 0, config));

        std::vector<ORB_SLAM2::orderedEdgePoint> nonFinite = {
            MakeDepthSample(0.00, 0.0, 1.0f),
            MakeDepthSample(0.01, 0.0, 1.0f),
            MakeDepthSample(0.02, 0.0, 1.0f),
            MakeDepthSample(0.03, 0.0, 1.0f),
            MakeDepthSample(
                0.04,
                0.0,
                std::numeric_limits<float>::quiet_NaN()),
            MakeDepthSample(0.05, 0.0, 1.0f),
            MakeDepthSample(0.06, 0.0, 1.0f),
            MakeDepthSample(0.07, 0.0, 1.0f),
            MakeDepthSample(0.08, 0.0, 1.0f)};
        const auto nonFiniteSegments =
            ORB_SLAM2::SplitContinuousCurveSamples(nonFinite, config);
        CHECK_TRUE(nonFiniteSegments.size() == 2);

        const std::vector<ORB_SLAM2::orderedEdgePoint> previousSource = {
            MakeDepthSample(0.00, 0.0, 1.0f),
            MakeDepthSample(0.01, 0.0, 1.0f),
            MakeDepthSample(0.02, 0.0, 1.0f),
            MakeDepthSample(0.03, 0.0, 1.0f)};
        const std::vector<ORB_SLAM2::orderedEdgePoint> currentSource = {
            MakeDepthSample(0.03, 0.0, 1.0f),
            MakeDepthSample(0.04, 0.0, 1.0f),
            MakeDepthSample(0.05, 0.0, 1.0f),
            MakeDepthSample(0.06, 0.0, 1.0f)};
        const auto previousSegments =
            ORB_SLAM2::SplitContinuousCurveSamples(previousSource, config);
        const auto currentSegments =
            ORB_SLAM2::SplitContinuousCurveSamples(currentSource, config);
        CHECK_TRUE(previousSegments.size() == 1);
        CHECK_TRUE(currentSegments.size() == 1);
        CHECK_TRUE(ORB_SLAM2::CanShareCurveContinuityGroup(
            previousSegments[0], 0, currentSegments[0], 1, config));

        std::vector<ORB_SLAM2::orderedEdgePoint> leadingInvalid = {
            MakeDepthSample(0.01, 0.0, 0.0f),
            MakeDepthSample(0.02, 0.0, 1.0f),
            MakeDepthSample(0.03, 0.0, 1.0f),
            MakeDepthSample(0.04, 0.0, 1.0f),
            MakeDepthSample(0.05, 0.0, 1.0f)};
        const auto leadingInvalidSegments =
            ORB_SLAM2::SplitContinuousCurveSamples(leadingInvalid, config);
        CHECK_TRUE(leadingInvalidSegments.size() == 1);
        CHECK_TRUE(!ORB_SLAM2::CanShareCurveContinuityGroup(
            previousSegments[0], 0, leadingInvalidSegments[0], 1, config));

        std::vector<ORB_SLAM2::orderedEdgePoint> trailingInvalid =
            previousSource;
        trailingInvalid.push_back(MakeDepthSample(0.02, 0.0, 0.0f));
        const auto trailingInvalidSegments =
            ORB_SLAM2::SplitContinuousCurveSamples(trailingInvalid, config);
        CHECK_TRUE(trailingInvalidSegments.size() == 1);
        CHECK_TRUE(!ORB_SLAM2::CanShareCurveContinuityGroup(
            trailingInvalidSegments[0], 0, currentSegments[0], 1, config));

        const std::vector<ORB_SLAM2::orderedEdgePoint> tooShort = {
            MakeDepthSample(0.00, 0.0, 1.0f),
            MakeDepthSample(0.01, 0.0, 1.0f),
            MakeDepthSample(0.02, 0.0, 1.0f)};
        CHECK_TRUE(ORB_SLAM2::SplitContinuousCurveSamples(
                       tooShort, config)
                       .empty());
    }

    void TestDepthLookupBounds(
        const std::shared_ptr<const ORB_SLAM2::CurveConfig> &config)
    {
        ORB_SLAM2::Frame frame;
        ORB_SLAM2::Frame::mnMinX = -1000.0f;
        ORB_SLAM2::Frame::mnMaxX = 1000.0f;
        ORB_SLAM2::Frame::mnMinY = -1000.0f;
        ORB_SLAM2::Frame::mnMaxY = 1000.0f;
        const cv::Mat depth = cv::Mat::ones(4, 4, CV_32F);
        std::vector<ORB_SLAM2::orderedEdgePoint> outsideImage = {
            ORB_SLAM2::orderedEdgePoint(-0.1, 1.0),
            ORB_SLAM2::orderedEdgePoint(-1.1, 1.0),
            ORB_SLAM2::orderedEdgePoint(4.0, 1.0),
            ORB_SLAM2::orderedEdgePoint(1.0, 4.0)};
        for (ORB_SLAM2::orderedEdgePoint &sample : outsideImage)
        {
            frame.assignProperty3DEach(sample, depth, config);
            CHECK_TRUE(sample.depth == 0.0f);
            CHECK_TRUE(sample.x_3d == 0.0);
            CHECK_TRUE(sample.y_3d == 0.0);
            CHECK_TRUE(sample.z_3d == 0.0);
        }
    }

    void TestGeometryQuality(const ORB_SLAM2::CurveConfig &config)
    {
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            MakeLine(3), config));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(MakeLine(12), config));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(MakeArc(12), config));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(MakeSingleCorner(), config));
        CHECK_TRUE(
            ORB_SLAM2::EvaluateCurveGeometry(
                MakeSingleCorner(), config)
                .highTurnRatio == 0.0);
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
            MakeCircle(0.03, 17), config));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
            MakeArc(96), config));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
            MakeSameDirectionTriangle(), config));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
            MakeSparseAlternatingCorners(), config));
        for (const std::size_t circleSampleCount :
             {9U, 17U, 33U, 65U})
        {
            CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
                MakeCircle(0.03, circleSampleCount), config));
        }

        const std::vector<cv::Point3d> zigzag = MakeZigzag(14);
        const ORB_SLAM2::CurveGeometryQuality zigzagQuality =
            ORB_SLAM2::EvaluateCurveGeometry(zigzag, config);
        const std::vector<cv::Point3d> arc = MakeArc(12);
        const ORB_SLAM2::CurveGeometryQuality arcQuality =
            ORB_SLAM2::EvaluateCurveGeometry(arc, config);
        for (const std::size_t factor : {1U, 2U, 4U, 8U})
        {
            const std::vector<cv::Point3d> denseZigzag =
                DensifyPolyline(zigzag, factor);
            CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
                denseZigzag, config));
            CheckEquivalentQuality(
                zigzagQuality,
                ORB_SLAM2::EvaluateCurveGeometry(denseZigzag, config));

            const std::vector<cv::Point3d> denseArc =
                DensifyPolyline(arc, factor);
            CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
                denseArc, config));
            CheckEquivalentQuality(
                arcQuality,
                ORB_SLAM2::EvaluateCurveGeometry(denseArc, config));
        }

        const std::vector<cv::Point3d> subSpacingZigzag =
            MakeZigzag(14, 0.006, 0.004);
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            subSpacingZigzag, config));
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            DensifyPolyline(subSpacingZigzag, 4), config));
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            MakeZigzag(5, 0.006, 0.004), config));
        std::vector<cv::Point3d> localizedZigzagWithLongTail =
            MakeZigzag(5, 0.006, 0.004);
        const cv::Point3d tailStart =
            localizedZigzagWithLongTail.back();
        for (std::size_t tailIndex = 1;
             tailIndex <= 50;
             ++tailIndex)
        {
            localizedZigzagWithLongTail.push_back(
                tailStart +
                cv::Point3d(0.02 * tailIndex, 0.0, 0.0));
        }
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            localizedZigzagWithLongTail, config));

        std::vector<cv::Point3d> localizedZigzagWithLowBendTail =
            MakeZigzag(5, 0.006, 0.004);
        cv::Point3d tailPoint =
            localizedZigzagWithLowBendTail.back();
        const cv::Point3d incomingTailDirection =
            tailPoint -
            localizedZigzagWithLowBendTail[
                localizedZigzagWithLowBendTail.size() - 2];
        const double baseHeading = std::atan2(
            incomingTailDirection.y,
            incomingTailDirection.x);
        const double lowBendRadians =
            20.0 * 3.14159265358979323846 / 180.0;
        for (std::size_t tailIndex = 0;
             tailIndex < 100;
             ++tailIndex)
        {
            const double heading =
                baseHeading +
                (tailIndex % 2 == 0 ? 0.0 : lowBendRadians);
            tailPoint += cv::Point3d(
                0.02 * std::cos(heading),
                0.02 * std::sin(heading),
                0.0);
            localizedZigzagWithLowBendTail.push_back(tailPoint);
        }
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            localizedZigzagWithLowBendTail, config));
        for (const std::size_t samplesPerPeriod :
             {4U, 8U, 16U, 32U, 64U})
        {
            CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
                MakeSineCurve(4, samplesPerPeriod), config));
        }

        const std::vector<cv::Point3d> repeatedHairpin = {
            cv::Point3d(0.00, 0.00000, 1.0),
            cv::Point3d(0.02, 0.00000, 1.0),
            cv::Point3d(0.00, 0.00000, 1.0),
            cv::Point3d(0.02, 0.00000, 1.0),
            cv::Point3d(0.00, 0.00000, 1.0)};
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            repeatedHairpin, config));

        const std::vector<cv::Point3d> nearHairpin = {
            cv::Point3d(0.00, 0.00000, 1.0),
            cv::Point3d(0.02, 0.00000, 1.0),
            cv::Point3d(0.00, 0.00003, 1.0),
            cv::Point3d(0.02, 0.00006, 1.0),
            cv::Point3d(0.00, 0.00009, 1.0)};
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            nearHairpin, config));

        std::vector<cv::Point3d> reversedZigzag = zigzag;
        std::reverse(reversedZigzag.begin(), reversedZigzag.end());
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(reversedZigzag, config));

        std::vector<cv::Point3d> nonFinite = MakeLine(12);
        nonFinite[5].x = std::numeric_limits<double>::infinity();
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(nonFinite, config));

        std::vector<cv::Point3d> discontinuous = MakeLine(12);
        for (std::size_t index = 6; index < discontinuous.size(); ++index)
            discontinuous[index].x += 0.3;
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(discontinuous, config));

        std::vector<cv::Point3d> duplicateLine;
        for (const cv::Point3d &point : MakeLine(12))
        {
            duplicateLine.push_back(point);
            duplicateLine.push_back(point);
        }
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
            duplicateLine, config));
        CHECK_TRUE(!ORB_SLAM2::IsCurveGeometryAcceptable(
            std::vector<cv::Point3d>(
                12, cv::Point3d(0.0, 0.0, 1.0)),
            config));
    }

    void TestFragmentControlPointRefit()
    {
        std::vector<ORB_SLAM2::orderedEdgePoint> firstSegment = {
            MakeDepthSample(0.0, 0.0, 1.0f),
            MakeDepthSample(2.5, 0.0, 1.0f),
            MakeDepthSample(5.0, 0.0, 1.0f),
            MakeDepthSample(7.5, 0.0, 1.0f),
            MakeDepthSample(10.0, 0.0, 1.0f)};

        ORB_SLAM2::BezierCurveFitter fitter(1.0);
        const ORB_SLAM2::BezierCurve fitted =
            fitter.fitSingleSegment(firstSegment, 0);
        CHECK_TRUE(fitted.controlPoints.size() >= 2);

        double parameter = 0.0;
        double distance = 0.0;
        cv::Point2d closestPoint;
        cv::Point2d normal;
        CHECK_TRUE(fitted.FindClosestPoint(
            cv::Point2d(35.0, 0.0),
            parameter,
            closestPoint,
            normal,
            distance));
        CHECK_TRUE(distance > 20.0);
    }

    void TestDepthGroupHardConflict()
    {
        ORB_SLAM2::BezierCurve firstDepthGroup;
        firstDepthGroup.edgeChainId = 7;
        firstDepthGroup.continuityGroupId = 0;

        ORB_SLAM2::BezierCurve bridgeFromAnotherEdge;
        bridgeFromAnotherEdge.edgeChainId = 8;
        bridgeFromAnotherEdge.continuityGroupId = 0;

        ORB_SLAM2::BezierCurve secondDepthGroup;
        secondDepthGroup.edgeChainId = 7;
        secondDepthGroup.continuityGroupId = 1;

        CHECK_TRUE(!firstDepthGroup.isDepthDisconnectedFrom(
            bridgeFromAnotherEdge));
        CHECK_TRUE(firstDepthGroup.isDepthDisconnectedFrom(
            secondDepthGroup));
    }

    bool SamePoints(
        const std::vector<cv::Point3d> &first,
        const std::vector<cv::Point3d> &second)
    {
        if (first.size() != second.size())
            return false;
        for (std::size_t index = 0; index < first.size(); ++index)
        {
            if (cv::norm(first[index] - second[index]) > 1e-12)
                return false;
        }
        return true;
    }

    void InitializeMinimalFrame(ORB_SLAM2::Frame &frame)
    {
        ORB_SLAM2::Frame::fx = 50.0f;
        ORB_SLAM2::Frame::fy = 50.0f;
        ORB_SLAM2::Frame::cx = 0.0f;
        ORB_SLAM2::Frame::cy = 0.0f;
        ORB_SLAM2::Frame::invfx = 0.02f;
        ORB_SLAM2::Frame::invfy = 0.02f;
        ORB_SLAM2::Frame::mnMinX = -1000.0f;
        ORB_SLAM2::Frame::mnMaxX = 1000.0f;
        ORB_SLAM2::Frame::mnMinY = -1000.0f;
        ORB_SLAM2::Frame::mnMaxY = 1000.0f;
        ORB_SLAM2::Frame::mfGridElementWidthInv = 1.0f;
        ORB_SLAM2::Frame::mfGridElementHeightInv = 1.0f;

        frame.mnId = 0;
        frame.mTimeStamp = 0.0;
        frame.mbf = 0.0f;
        frame.mb = 0.0f;
        frame.mThDepth = 10.0f;
        frame.N = 0;
        frame.NC = 0;
        frame.mnScaleLevels = 1;
        frame.mfScaleFactor = 1.0f;
        frame.mfLogScaleFactor = 0.0f;
        frame.mK = cv::Mat::eye(3, 3, CV_32F);
        frame.SetPose(cv::Mat::eye(4, 4, CV_32F));
    }

    void TestTransactionalMapCurveMerge(
        const std::shared_ptr<const ORB_SLAM2::CurveConfig> &config)
    {
        ORB_SLAM2::Map map;
        ORB_SLAM2::Frame frame;
        InitializeMinimalFrame(frame);
        ORB_SLAM2::KeyFrame keyFrame(frame, &map, NULL);

        const std::vector<cv::Point3d> straight = MakeLine(14);
        std::vector<cv::Point3d> alternatingCandidate = straight;
        for (std::size_t index = 0; index < alternatingCandidate.size(); ++index)
            alternatingCandidate[index].y = index % 2 == 0 ? 0.08 : -0.08;
        ORB_SLAM2::MapCurve survivor(straight, &keyFrame, &map);
        ORB_SLAM2::MapCurve noisyCandidate(alternatingCandidate, &keyFrame, &map);
        const std::vector<cv::Point3d> beforeRejectedMerge =
            survivor.GetCurvePoints();

        CHECK_TRUE(!survivor.MergeGeometryFrom(&noisyCandidate, config));
        CHECK_TRUE(SamePoints(beforeRejectedMerge, survivor.GetCurvePoints()));

        std::vector<cv::Point3d> parallel = straight;
        for (cv::Point3d &point : parallel)
            point.y += 0.02;
        ORB_SLAM2::MapCurve smoothSurvivor(straight, &keyFrame, &map);
        ORB_SLAM2::MapCurve smoothCandidate(parallel, &keyFrame, &map);
        CHECK_TRUE(smoothSurvivor.MergeGeometryFrom(&smoothCandidate, config));
        CHECK_TRUE(!SamePoints(straight, smoothSurvivor.GetCurvePoints()));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
            smoothSurvivor.GetCurvePoints(), *config));
    }

    void SetObservedCurve(
        ORB_SLAM2::Frame &frame,
        const std::vector<cv::Point3d> &cameraPoints)
    {
        ORB_SLAM2::BezierCurve curve;
        curve.sampledPoints.reserve(cameraPoints.size());
        for (const cv::Point3d &cameraPoint : cameraPoints)
        {
            ORB_SLAM2::orderedEdgePoint sample(
                ORB_SLAM2::Frame::fx * cameraPoint.x / cameraPoint.z +
                    ORB_SLAM2::Frame::cx,
                ORB_SLAM2::Frame::fy * cameraPoint.y / cameraPoint.z +
                    ORB_SLAM2::Frame::cy);
            sample.depth = static_cast<float>(cameraPoint.z);
            sample.x_3d = cameraPoint.x;
            sample.y_3d = cameraPoint.y;
            sample.z_3d = cameraPoint.z;
            curve.sampledPoints.push_back(sample);
        }
        frame.mvBezierCurves.assign(1, curve);
    }

    void TestTransactionalMapCurveExtension(
        const std::shared_ptr<const ORB_SLAM2::CurveConfig> &config)
    {
        ORB_SLAM2::Map map;
        ORB_SLAM2::Frame frame;
        InitializeMinimalFrame(frame);
        ORB_SLAM2::KeyFrame keyFrame(frame, &map, NULL);

        const std::vector<cv::Point3d> straight = MakeLine(14);
        std::vector<cv::Point3d> alternatingObservation = straight;
        for (std::size_t index = 0; index < alternatingObservation.size(); ++index)
            alternatingObservation[index].y = index % 2 == 0 ? 0.08 : -0.08;

        ORB_SLAM2::MapCurve rejectedCurve(straight, &keyFrame, &map);
        const std::vector<cv::Point3d> beforeRejectedExtension =
            rejectedCurve.GetCurvePoints();
        SetObservedCurve(frame, alternatingObservation);
        rejectedCurve.ExtendWithObservation(frame, 0, config);
        CHECK_TRUE(SamePoints(
            beforeRejectedExtension, rejectedCurve.GetCurvePoints()));

        std::vector<cv::Point3d> parallelObservation = straight;
        for (cv::Point3d &point : parallelObservation)
            point.y += 0.02;
        ORB_SLAM2::MapCurve acceptedCurve(straight, &keyFrame, &map);
        SetObservedCurve(frame, parallelObservation);
        acceptedCurve.ExtendWithObservation(frame, 0, config);
        CHECK_TRUE(!SamePoints(straight, acceptedCurve.GetCurvePoints()));
        CHECK_TRUE(ORB_SLAM2::IsCurveGeometryAcceptable(
            acceptedCurve.GetCurvePoints(), *config));
    }

    void TestLifecycleAssociation()
    {
        ORB_SLAM2::Map map;
        ORB_SLAM2::Frame frame;
        InitializeMinimalFrame(frame);
        frame.NC = 1;
        frame.mvpMapCurves.assign(
            1, static_cast<ORB_SLAM2::MapCurve *>(NULL));
        ORB_SLAM2::KeyFrame keyFrame(frame, &map, NULL);
        ORB_SLAM2::MapCurve curve(MakeLine(12), &keyFrame, &map);

        CHECK_TRUE(!curve.AssociateWithKeyFrame(&keyFrame, 1));
        CHECK_TRUE(curve.Observations() == 0);
        CHECK_TRUE(keyFrame.GetMapCurve(0) == NULL);

        std::atomic<bool> start(false);
        std::thread associateThread([&]()
                                    {
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            curve.AssociateWithKeyFrame(&keyFrame, 0); });
        std::thread cullingThread([&]()
                                 {
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            curve.SetBadFlag(); });
        start.store(true, std::memory_order_release);
        associateThread.join();
        cullingThread.join();

        CHECK_TRUE(curve.isBad());
        CHECK_TRUE(curve.Observations() == 0);
        CHECK_TRUE(keyFrame.GetMapCurve(0) == NULL);
        CHECK_TRUE(!curve.AssociateWithKeyFrame(&keyFrame, 0));
        CHECK_TRUE(keyFrame.GetMapCurve(0) == NULL);
    }

    void TestEraseAssociateLifecycleRace()
    {
        ORB_SLAM2::Map map;
        ORB_SLAM2::Frame firstFrame;
        ORB_SLAM2::Frame secondFrame;
        ORB_SLAM2::Frame thirdFrame;
        InitializeMinimalFrame(firstFrame);
        InitializeMinimalFrame(secondFrame);
        InitializeMinimalFrame(thirdFrame);
        for (ORB_SLAM2::Frame *frame :
             {&firstFrame, &secondFrame, &thirdFrame})
        {
            frame->NC = 1;
            frame->mvpMapCurves.assign(
                1, static_cast<ORB_SLAM2::MapCurve *>(NULL));
        }
        ORB_SLAM2::KeyFrame firstKeyFrame(firstFrame, &map, NULL);
        ORB_SLAM2::KeyFrame secondKeyFrame(secondFrame, &map, NULL);
        ORB_SLAM2::KeyFrame thirdKeyFrame(thirdFrame, &map, NULL);
        ORB_SLAM2::MapCurve curve(
            MakeLine(12), &firstKeyFrame, &map);
        CHECK_TRUE(curve.AssociateWithKeyFrame(
            &firstKeyFrame, 0));
        CHECK_TRUE(curve.AssociateWithKeyFrame(
            &secondKeyFrame, 0));

        std::atomic<bool> start(false);
        std::thread eraseThread([&]()
                                {
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            curve.EraseObservation(&secondKeyFrame); });
        std::thread associateThread([&]()
                                    {
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            curve.AssociateWithKeyFrame(&thirdKeyFrame, 0); });
        start.store(true, std::memory_order_release);
        eraseThread.join();
        associateThread.join();

        if (curve.isBad())
        {
            CHECK_TRUE(curve.Observations() == 0);
            CHECK_TRUE(thirdKeyFrame.GetMapCurve(0) == NULL);
        }
        else
        {
            CHECK_TRUE(curve.Observations() == 2);
            CHECK_TRUE(curve.IsInKeyFrame(&firstKeyFrame));
            CHECK_TRUE(curve.IsInKeyFrame(&thirdKeyFrame));
            CHECK_TRUE(secondKeyFrame.GetMapCurve(0) == NULL);
            CHECK_TRUE(
                thirdKeyFrame.GetMapCurve(0) == &curve);
            curve.SetBadFlag();
        }
        CHECK_TRUE(secondKeyFrame.GetMapCurve(0) == NULL);
    }

    void TestMapCurveReplacementAssociations(
        const std::shared_ptr<const ORB_SLAM2::CurveConfig> &config)
    {
        ORB_SLAM2::Map map;
        ORB_SLAM2::Frame firstFrame;
        ORB_SLAM2::Frame secondFrame;
        InitializeMinimalFrame(firstFrame);
        InitializeMinimalFrame(secondFrame);
        for (ORB_SLAM2::Frame *frame : {&firstFrame, &secondFrame})
        {
            frame->NC = 1;
            frame->mvpMapCurves.assign(
                1, static_cast<ORB_SLAM2::MapCurve *>(NULL));
        }

        ORB_SLAM2::KeyFrame firstKeyFrame(firstFrame, &map, NULL);
        ORB_SLAM2::KeyFrame secondKeyFrame(secondFrame, &map, NULL);
        const std::vector<cv::Point3d> straight = MakeLine(12);
        std::vector<cv::Point3d> parallel = straight;
        for (cv::Point3d &point : parallel)
            point.y += 0.02;

        ORB_SLAM2::MapCurve survivor(straight, &firstKeyFrame, &map);
        ORB_SLAM2::MapCurve discarded(parallel, &secondKeyFrame, &map);
        CHECK_TRUE(survivor.AssociateWithKeyFrame(&firstKeyFrame, 0));
        CHECK_TRUE(discarded.AssociateWithKeyFrame(&secondKeyFrame, 0));

        CHECK_TRUE(discarded.Replace(&survivor, config));
        CHECK_TRUE(discarded.isBad());
        CHECK_TRUE(discarded.GetReplaced() == &survivor);
        CHECK_TRUE(firstKeyFrame.GetMapCurve(0) == &survivor);
        CHECK_TRUE(secondKeyFrame.GetMapCurve(0) == &survivor);
        CHECK_TRUE(survivor.Observations() == 2);
        CHECK_TRUE(survivor.IsInKeyFrame(&firstKeyFrame));
        CHECK_TRUE(survivor.IsInKeyFrame(&secondKeyFrame));

        // A cleanup operation that still expects the discarded curve must
        // never erase the survivor installed by Replace().
        CHECK_TRUE(!secondKeyFrame.ReplaceMapCurveMatch(
            0, &discarded, static_cast<ORB_SLAM2::MapCurve *>(NULL)));
        CHECK_TRUE(secondKeyFrame.GetMapCurve(0) == &survivor);
        CHECK_TRUE(survivor.Observations() == 2);
        survivor.SetBadFlag();
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: curve_geometry_tests <settings.yaml>\n";
        return 2;
    }

    const std::shared_ptr<const ORB_SLAM2::CurveConfig> config =
        std::make_shared<ORB_SLAM2::CurveConfig>(std::string(argv[1]), true);
    TestDepthSegmentation(*config);
    TestDepthLookupBounds(config);
    TestGeometryQuality(*config);
    TestFragmentControlPointRefit();
    TestDepthGroupHardConflict();
    TestTransactionalMapCurveMerge(config);
    TestTransactionalMapCurveExtension(config);
    TestLifecycleAssociation();
    TestEraseAssociateLifecycleRace();
    TestMapCurveReplacementAssociations(config);

    if (gFailureCount != 0)
    {
        std::cerr << gFailureCount << " curve geometry checks failed\n";
        return 1;
    }
    std::cout << "curve geometry checks passed\n";
    return 0;
}
