#include "Curve/MapCurve.h"

namespace ORB_SLAM2
{
    long unsigned int MapCurve::nNextId = 0;
    std::mutex MapCurve::mGlobalMutex;

    MapCurve::MapCurve(std::vector<cv::Point3d> curvePoints, KeyFrame *pReferenceKF, Map *pMap) : mnFirstKFid(pReferenceKF->mnId), mnFirstFrame(pReferenceKF->mnFrameId), nObs(0), mpRefKF(pReferenceKF), mnVisible(1), mnFound(1), mbBad(false), mpMap(pMap)
    {
        mCurvePoints = curvePoints;

        // MapCurves can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    // 在关键帧pKF中的第idx个曲线特征
    void MapCurve::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;

        nObs++;
    }

    MapCurve *MapCurve::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }
}
