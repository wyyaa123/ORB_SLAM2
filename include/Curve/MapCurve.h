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
        size_t ExtendWithObservation(const Frame &frame, size_t observedCurveIndex);

        std::map<KeyFrame *, size_t> GetObservations();
        int Observations();

        void AddObservation(KeyFrame *pKF, size_t idx);
        void EraseObservation(KeyFrame *pKF);

        int GetIndexInKeyFrame(KeyFrame *pKF);
        bool IsInKeyFrame(KeyFrame *pKF);

        void SetBadFlag();
        bool isBad();

        MapCurve *GetReplaced();

        void IncreaseVisible(int n = 1);
        void IncreaseFound(int n = 1);
        float GetFoundRatio();

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        bool mbTrackInView;
        long unsigned int mnTrackReferenceForFrame;
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

    private:
        static bool ProjectWorldPoint(const cv::Point3d &worldPoint, const Frame &frame, cv::Point2d &imagePoint);
        static size_t FindNearestPointIndex(const cv::Point2d &queryPoint, const std::vector<cv::Point2d> &points);
    };
}

#endif
