#ifndef MAPCURVE_H
#define MAPCURVE_H

#include <opencv2/core/core.hpp>
#include <map>
#include <mutex>
#include <vector>

namespace ORB_SLAM2
{
    class KeyFrame;
    class Map;
    class Frame;

    class MapCurve
    {
    public:
        MapCurve(std::vector<cv::Point3d> curvePoints, Map *pMap, KeyFrame *pReferenceKF = NULL);
    };
}

#endif
