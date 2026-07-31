#ifndef CURVEOBSERVATION_H
#define CURVEOBSERVATION_H

#include <cstddef>
#include <opencv2/core/core.hpp>
#include <string>

namespace ORB_SLAM2
{
    class MapCurve;

    struct CurveSampleCorrespondence
    {
        CurveSampleCorrespondence();

        MapCurve *pMapCurve;
        cv::Point3d worldPoint;
        cv::Point2d projectedPoint;
        cv::Point2d observedPoint;
        cv::Point2d observedNormal;
        std::size_t observedCurveIndex;
        double observedParameter;
        double initialDistance;
        double normalizedWeight;
    };

    struct CurveMatchDiagnostic
    {
        CurveMatchDiagnostic();

        MapCurve *pMapCurve;
        bool matched;
        std::string failureType;
        std::string failureDetail;
    };
}

#endif // CURVEOBSERVATION_H
