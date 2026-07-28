/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Curve/MapCurve.h"
#include "Map.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <mutex>

namespace ORB_SLAM2
{

    class Tracking;
    class Viewer;

    class FrameDrawer
    {
    public:
        FrameDrawer(Map *pMap);

        // Update info from the last processed frame.
        void Update(Tracking *pTracker);

        // Draw last processed frame.
        cv::Mat DrawFrame();

        // Draw the current curves in the frame.
        cv::Mat DrawFrameCurves();
        cv::Mat DrawCurveAssociations();

    protected:
        void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);
        cv::Scalar AssociationColor(std::size_t colorIndex);
        void DrawSampledCurve(cv::Mat &image, const std::vector<cv::Point2f> &samples, const cv::Scalar &color, int thickness);
        bool ProjectWorldPoint(const cv::Point3d &worldPoint, const cv::Mat &Tcw, cv::Point2f &imagePoint);
        void AppendResampledSegment(const cv::Point2f &first, const cv::Point2f &second, std::vector<cv::Point2f> &samples);
        bool ProjectMapCurve(MapCurve *pMapCurve, const cv::Mat &Tcw, float margin, std::vector<cv::Point2f> &projectedSamples);

        // Info of the frame to be drawn
        cv::Mat mIm;
        cv::Mat mCurveAssociationTcw;
        vector<BezierCurve> mvCurveAssociationCurves;
        vector<MapCurve *> mvpCurveAssociationMatches;
        vector<MapCurve *> mvpCurveAssociationCandidates;
        vector<CurveMatchDiagnostic> mvCurveMatchDiagnostics;
        int N;
        vector<cv::KeyPoint> mvCurrentKeys;
        vector<BezierCurve> mpCurrentCurves;
        vector<bool> mvbMap, mvbVO;
        bool mbOnlyTracking;
        int mnTracked, mnTrackedVO;
        vector<cv::KeyPoint> mvIniKeys;
        vector<int> mvIniMatches;
        int mState;

        Map *mpMap;

        std::mutex mMutex;
    };

} // namespace ORB_SLAM

#endif // FRAMEDRAWER_H
