#include "Curve/MapCurve.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ORB_SLAM2
{
    long unsigned int MapCurve::nNextId = 0;
    std::mutex MapCurve::mGlobalMutex;

    MapCurve::MapCurve(std::vector<cv::Point3d> curvePoints, KeyFrame *pReferenceKF, Map *pMap) : mnFirstKFid(pReferenceKF->mnId), mnFirstFrame(pReferenceKF->mnFrameId), nObs(0), mnLastFrameSeen(0), mpRefKF(pReferenceKF), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
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

    std::vector<cv::Point3d> MapCurve::GetCurvePoints()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mCurvePoints;
    }

    size_t MapCurve::ExtendWithObservation(const std::vector<cv::Point3d> &observedPoints, const double maximumAssociationDistance, const size_t minimumOverlapPoints)
    {
        if (observedPoints.size() < 2 || maximumAssociationDistance <= 0.0)
            return 0;

        unique_lock<mutex> lock(mMutexPos);
        if (mCurvePoints.size() < 2)
        {
            mCurvePoints = observedPoints;
            return observedPoints.size();
        }

        const double maximumDistanceSquared = maximumAssociationDistance * maximumAssociationDistance;
        std::vector<cv::Point3d> orientedObservation = observedPoints;
        std::vector<size_t> nearestMapIndices;
        std::vector<double> nearestMapDistancesSquared;
        const auto computeNearestMapPoints = [this, &nearestMapIndices, &nearestMapDistancesSquared](const std::vector<cv::Point3d> &observation)
        {
            nearestMapIndices.assign(observation.size(), 0);
            nearestMapDistancesSquared.assign(observation.size(), std::numeric_limits<double>::max());
            for (size_t observationIndex = 0; observationIndex < observation.size(); ++observationIndex)
            {
                for (size_t mapIndex = 0; mapIndex < mCurvePoints.size(); ++mapIndex)
                {
                    const cv::Point3d delta = observation[observationIndex] - mCurvePoints[mapIndex];
                    const double distanceSquared = delta.dot(delta);
                    if (distanceSquared < nearestMapDistancesSquared[observationIndex])
                    {
                        nearestMapDistancesSquared[observationIndex] = distanceSquared;
                        nearestMapIndices[observationIndex] = mapIndex;
                    }
                }
            }
        };

        computeNearestMapPoints(orientedObservation);
        size_t firstOverlapIndex = orientedObservation.size();
        size_t lastOverlapIndex = 0;
        size_t overlapCount = 0;
        for (size_t observationIndex = 0; observationIndex < orientedObservation.size(); ++observationIndex)
        {
            if (nearestMapDistancesSquared[observationIndex] <= maximumDistanceSquared)
            {
                firstOverlapIndex = std::min(firstOverlapIndex, observationIndex);
                lastOverlapIndex = observationIndex;
                ++overlapCount;
            }
        }
        if (overlapCount < std::max<size_t>(2, minimumOverlapPoints) || firstOverlapIndex >= orientedObservation.size())
            return 0;

        if (nearestMapIndices[firstOverlapIndex] > nearestMapIndices[lastOverlapIndex])
        {
            std::reverse(orientedObservation.begin(), orientedObservation.end());
            computeNearestMapPoints(orientedObservation);
            firstOverlapIndex = orientedObservation.size();
            lastOverlapIndex = 0;
            overlapCount = 0;
            for (size_t observationIndex = 0; observationIndex < orientedObservation.size(); ++observationIndex)
            {
                if (nearestMapDistancesSquared[observationIndex] <= maximumDistanceSquared)
                {
                    firstOverlapIndex = std::min(firstOverlapIndex, observationIndex);
                    lastOverlapIndex = observationIndex;
                    ++overlapCount;
                }
            }
        }
        if (overlapCount < std::max<size_t>(2, minimumOverlapPoints) || firstOverlapIndex >= orientedObservation.size() || nearestMapIndices[firstOverlapIndex] > nearestMapIndices[lastOverlapIndex])
            return 0;

        const size_t edgeTolerance = std::max<size_t>(2, mCurvePoints.size() / 10);
        bool reachesMapFront = nearestMapIndices[firstOverlapIndex] <= edgeTolerance && firstOverlapIndex > 0;
        bool reachesMapBack = nearestMapIndices[lastOverlapIndex] + edgeTolerance >= mCurvePoints.size() - 1 && lastOverlapIndex + 1 < orientedObservation.size();
        if (reachesMapFront)
        {
            const cv::Point3d mapDirection = mCurvePoints.front() - mCurvePoints[1];
            const cv::Point3d observationDirection = orientedObservation[firstOverlapIndex - 1] - orientedObservation[firstOverlapIndex];
            const double directionLengths = cv::norm(mapDirection) * cv::norm(observationDirection);
            reachesMapFront = directionLengths > 1e-9 && mapDirection.dot(observationDirection) / directionLengths >= 0.5;
        }
        if (reachesMapBack)
        {
            const cv::Point3d mapDirection = mCurvePoints.back() - mCurvePoints[mCurvePoints.size() - 2];
            const cv::Point3d observationDirection = orientedObservation[lastOverlapIndex + 1] - orientedObservation[lastOverlapIndex];
            const double directionLengths = cv::norm(mapDirection) * cv::norm(observationDirection);
            reachesMapBack = directionLengths > 1e-9 && mapDirection.dot(observationDirection) / directionLengths >= 0.5;
        }
        const double maximumExtensionGap = 3.0 * maximumAssociationDistance;
        size_t prefixBegin = firstOverlapIndex;
        if (reachesMapFront)
        {
            while (prefixBegin > 0 && cv::norm(orientedObservation[prefixBegin] - orientedObservation[prefixBegin - 1]) <= maximumExtensionGap)
                --prefixBegin;
        }
        size_t suffixEnd = lastOverlapIndex;
        if (reachesMapBack)
        {
            while (suffixEnd + 1 < orientedObservation.size() && cv::norm(orientedObservation[suffixEnd + 1] - orientedObservation[suffixEnd]) <= maximumExtensionGap)
                ++suffixEnd;
        }

        const size_t prefixCount = reachesMapFront ? firstOverlapIndex - prefixBegin : 0;
        const size_t suffixCount = reachesMapBack ? suffixEnd - lastOverlapIndex : 0;
        if (prefixCount == 0 && suffixCount == 0)
            return 0;

        std::vector<cv::Point3d> extendedCurve;
        extendedCurve.reserve(prefixCount + mCurvePoints.size() + suffixCount);
        if (prefixCount > 0)
            extendedCurve.insert(extendedCurve.end(), orientedObservation.begin() + prefixBegin, orientedObservation.begin() + firstOverlapIndex);
        extendedCurve.insert(extendedCurve.end(), mCurvePoints.begin(), mCurvePoints.end());
        if (suffixCount > 0)
            extendedCurve.insert(extendedCurve.end(), orientedObservation.begin() + lastOverlapIndex + 1, orientedObservation.begin() + suffixEnd + 1);
        mCurvePoints.swap(extendedCurve);
        return prefixCount + suffixCount;
    }

    int MapCurve::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    bool MapCurve::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mbBad;
    }
}
