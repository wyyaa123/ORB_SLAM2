#include "Curve/MapCurve.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ORB_SLAM2
{
    long unsigned int MapCurve::nNextId = 0;
    std::mutex MapCurve::mGlobalMutex;

    MapCurve::MapCurve(std::vector<cv::Point3d> curvePoints, KeyFrame *pReferenceKF, Map *pMap) : mnFirstKFid(pReferenceKF->mnId), mnFirstFrame(pReferenceKF->mnFrameId), nObs(0), mbTrackInView(false), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mpRefKF(pReferenceKF), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
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

    void MapCurve::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF))
            {
                nObs--;

                mObservations.erase(pKF);

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                if (nObs < 2)
                    bBad = true;
            }
        }

        if (bBad)
            SetBadFlag();
    }

    MapCurve *MapCurve::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapCurve::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapCurve::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapCurve::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    std::vector<cv::Point3d> MapCurve::GetCurvePoints()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mCurvePoints;
    }

    bool MapCurve::ProjectWorldPoint(const cv::Point3d &worldPoint, const Frame &frame, cv::Point2d &imagePoint)
    {
        if (frame.mTcw.empty())
            return false;

        const double cameraX = frame.mTcw.at<float>(0, 0) * worldPoint.x + frame.mTcw.at<float>(0, 1) * worldPoint.y + frame.mTcw.at<float>(0, 2) * worldPoint.z + frame.mTcw.at<float>(0, 3);
        const double cameraY = frame.mTcw.at<float>(1, 0) * worldPoint.x + frame.mTcw.at<float>(1, 1) * worldPoint.y + frame.mTcw.at<float>(1, 2) * worldPoint.z + frame.mTcw.at<float>(1, 3);
        const double cameraZ = frame.mTcw.at<float>(2, 0) * worldPoint.x + frame.mTcw.at<float>(2, 1) * worldPoint.y + frame.mTcw.at<float>(2, 2) * worldPoint.z + frame.mTcw.at<float>(2, 3);
        if (cameraZ <= 0.0)
            return false;

        imagePoint.x = frame.fx * cameraX / cameraZ + frame.cx;
        imagePoint.y = frame.fy * cameraY / cameraZ + frame.cy;
        return imagePoint.x >= frame.mnMinX && imagePoint.x <= frame.mnMaxX && imagePoint.y >= frame.mnMinY && imagePoint.y <= frame.mnMaxY;
    }

    size_t MapCurve::FindNearestPointIndex(const cv::Point2d &queryPoint, const std::vector<cv::Point2d> &points)
    {
        if (points.empty())
            return 0;

        size_t nearestIndex = 0;
        double nearestDistanceSquared = std::numeric_limits<double>::max();
        for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
        {
            const cv::Point2d difference = points[pointIndex] - queryPoint;
            const double distanceSquared = difference.dot(difference);
            if (distanceSquared < nearestDistanceSquared)
            {
                nearestDistanceSquared = distanceSquared;
                nearestIndex = pointIndex;
            }
        }
        return nearestIndex;
    }

    size_t MapCurve::ExtendWithObservation(const Frame &frame, const size_t observedCurveIndex)
    {
        if (observedCurveIndex >= frame.mvBezierCurves.size() || frame.mTcw.empty())
            return 0;

        const BezierCurve &observedCurve = frame.mvBezierCurves[observedCurveIndex];
        const std::vector<cv::Point3d> observedWorldPoints = frame.UnprojectCurve(static_cast<int>(observedCurveIndex));
        std::vector<cv::Point2d> observedImagePoints;
        observedImagePoints.reserve(observedCurve.sampledPoints.size());
        for (const orderedEdgePoint &point : observedCurve.sampledPoints)
        {
            if (point.depth > 0.0f)
                observedImagePoints.push_back(cv::Point2d(point.x, point.y));
        }
        if (observedWorldPoints.size() < 2 || observedWorldPoints.size() != observedImagePoints.size())
            return 0;

        unique_lock<mutex> lock(mMutexPos);
        if (mCurvePoints.size() < 2)
        {
            mCurvePoints = observedWorldPoints;
            return observedWorldPoints.size();
        }

        std::vector<cv::Point2d> projectedMapPoints;
        std::vector<size_t> projectedMapIndices;
        projectedMapPoints.reserve(mCurvePoints.size());
        projectedMapIndices.reserve(mCurvePoints.size());
        for (size_t mapPointIndex = 0; mapPointIndex < mCurvePoints.size(); ++mapPointIndex)
        {
            cv::Point2d projectedPoint;
            if (!ProjectWorldPoint(mCurvePoints[mapPointIndex], frame, projectedPoint) || projectedPoint.x < frame.mnMinX || projectedPoint.x > frame.mnMaxX || projectedPoint.y < frame.mnMinY || projectedPoint.y > frame.mnMaxY)
                continue;
            projectedMapPoints.push_back(projectedPoint);
            projectedMapIndices.push_back(mapPointIndex);
        }
        if (projectedMapPoints.size() < 2)
            return 0;

        std::vector<cv::Point2d> orientedImagePoints = observedImagePoints;
        std::vector<cv::Point3d> orientedWorldPoints = observedWorldPoints;
        // Make the current observation follow the same front-to-back order as the projected map curve.
        const double forwardEndpointCost = cv::norm(orientedImagePoints.front() - projectedMapPoints.front()) + cv::norm(orientedImagePoints.back() - projectedMapPoints.back());
        const double reverseEndpointCost = cv::norm(orientedImagePoints.front() - projectedMapPoints.back()) + cv::norm(orientedImagePoints.back() - projectedMapPoints.front());
        if (reverseEndpointCost < forwardEndpointCost)
        {
            std::reverse(orientedImagePoints.begin(), orientedImagePoints.end());
            std::reverse(orientedWorldPoints.begin(), orientedWorldPoints.end());
        }

        size_t firstOverlapIndex = orientedImagePoints.size();
        size_t lastOverlapIndex = 0;
        // The nearest observed indices covered by all visible map samples define the overlap interval.
        for (const cv::Point2d &projectedMapPoint : projectedMapPoints)
        {
            const size_t nearestObservedIndex = FindNearestPointIndex(projectedMapPoint, orientedImagePoints);
            firstOverlapIndex = std::min(firstOverlapIndex, nearestObservedIndex);
            lastOverlapIndex = std::max(lastOverlapIndex, nearestObservedIndex);
        }
        if (firstOverlapIndex >= orientedImagePoints.size())
            return 0;

        const bool mapFrontIsVisible = projectedMapIndices.front() == 0;
        const bool mapBackIsVisible = projectedMapIndices.back() + 1 == mCurvePoints.size();
        const size_t prefixCount = mapFrontIsVisible ? firstOverlapIndex : 0;
        const size_t suffixCount = mapBackIsVisible && lastOverlapIndex + 1 < orientedWorldPoints.size() ? orientedWorldPoints.size() - lastOverlapIndex - 1 : 0;
        if (prefixCount == 0 && suffixCount == 0)
            return 0;

        std::vector<cv::Point3d> extendedCurve;
        extendedCurve.reserve(prefixCount + mCurvePoints.size() + suffixCount);
        if (prefixCount > 0)
            extendedCurve.insert(extendedCurve.end(), orientedWorldPoints.begin(), orientedWorldPoints.begin() + firstOverlapIndex);
        // Retain the existing map geometry in the overlap and append only the observation outside both ends.
        extendedCurve.insert(extendedCurve.end(), mCurvePoints.begin(), mCurvePoints.end());
        if (suffixCount > 0)
            extendedCurve.insert(extendedCurve.end(), orientedWorldPoints.begin() + lastOverlapIndex + 1, orientedWorldPoints.end());
        mCurvePoints.swap(extendedCurve);
        return prefixCount + suffixCount;
    }

    std::map<KeyFrame *, size_t> MapCurve::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapCurve::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    bool ORB_SLAM2::MapCurve::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations.count(pKF);
    }

    void ORB_SLAM2::MapCurve::SetBadFlag()
    {
        map<KeyFrame *, size_t> observations;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            observations = mObservations;
            mObservations.clear();
        }
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            pKF->EraseMapCurveMatch(mit->second);
        }

        mpMap->EraseMapCurve(this);
    }

    bool MapCurve::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mbBad;
    }
}
