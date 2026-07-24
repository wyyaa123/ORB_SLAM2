#ifndef MAPCURVE_H
#define MAPCURVE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

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
        MapCurve(std::vector<cv::Point3d> curvePoints, KeyFrame *pReferenceKF, Map *pMap);

        std::vector<cv::Point3d> GetCurvePoints();
        size_t ExtendWithObservation(const std::vector<cv::Point3d> &observedPoints, double maximumAssociationDistance, size_t minimumOverlapPoints);

        int Observations();

        void AddObservation(KeyFrame *pKF, size_t idx);

        MapCurve *GetReplaced();

        bool isBad();

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        bool mbTrackInView;
        long unsigned int mnLastFrameSeen;

        static std::mutex mGlobalMutex;

    protected:
        std::vector<cv::Point3d> mCurvePoints;

        std::map<KeyFrame *, size_t> mObservations;

        KeyFrame *mpRefKF;

        int mnVisible;
        int mnFound;
        
        bool mbBad;
        MapCurve *mpReplaced;

        Map *mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };
}

#endif
