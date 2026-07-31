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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ORB_SLAM2
{
    namespace
    {
        void DrawDashedLine(cv::Mat &image, const cv::Point2d &first, const cv::Point2d &second,
                            const cv::Scalar &color, const int thickness,
                            const double dashLength = 6.0, const double gapLength = 4.0)
        {
            if (!std::isfinite(first.x) || !std::isfinite(first.y) ||
                !std::isfinite(second.x) || !std::isfinite(second.y))
                return;

            const cv::Point2d delta = second - first;
            const double length = cv::norm(delta);
            if (length <= 1e-6)
                return;

            const cv::Point2d direction = delta * (1.0 / length);
            for (double offset = 0.0; offset < length; offset += dashLength + gapLength)
            {
                const cv::Point2d dashBegin = first + direction * offset;
                const cv::Point2d dashEnd = first + direction * std::min(length, offset + dashLength);
                cv::line(image,
                         cv::Point(cvRound(dashBegin.x), cvRound(dashBegin.y)),
                         cv::Point(cvRound(dashEnd.x), cvRound(dashEnd.y)),
                         color, thickness, cv::LINE_AA);
            }
        }
    }

    FrameDrawer::FrameDrawer(Map *pMap) : mpMap(pMap)
    {
        mState = Tracking::SYSTEM_NOT_READY;
        mIm = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat im;
        vector<cv::KeyPoint> vIniKeys;     // Initialization: KeyPoints in reference frame
        vector<int> vMatches;              // Initialization: correspondeces with reference keypoints
        vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
        vector<bool> vbVO, vbMap;          // Tracked MapPoints in current frame
        int state;                         // Tracking state

        // Copy variables within scoped mutex
        {
            unique_lock<mutex> lock(mMutex);
            state = mState;
            if (mState == Tracking::SYSTEM_NOT_READY)
                mState = Tracking::NO_IMAGES_YET;

            mIm.copyTo(im);

            if (mState == Tracking::NOT_INITIALIZED)
            {
                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
            }
            else if (mState == Tracking::OK)
            {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
            }
            else if (mState == Tracking::LOST)
            {
                vCurrentKeys = mvCurrentKeys;
            }
        } // destroy scoped mutex -> release mutex

        if (im.channels() < 3) // this should be always true
            cvtColor(im, im, CV_GRAY2BGR);

        // Draw
        if (state == Tracking::NOT_INITIALIZED) // INITIALIZING
        {
            for (unsigned int i = 0; i < vMatches.size(); i++)
            {
                if (vMatches[i] >= 0)
                {
                    cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt,
                             cv::Scalar(0, 255, 0));
                }
            }
        }
        else if (state == Tracking::OK) // TRACKING
        {
            mnTracked = 0;
            mnTrackedVO = 0;
            const float r = 5;
            const int n = vCurrentKeys.size();
            for (int i = 0; i < n; i++)
            {
                if (vbVO[i] || vbMap[i])
                {
                    cv::Point2f pt1, pt2;
                    pt1.x = vCurrentKeys[i].pt.x - r;
                    pt1.y = vCurrentKeys[i].pt.y - r;
                    pt2.x = vCurrentKeys[i].pt.x + r;
                    pt2.y = vCurrentKeys[i].pt.y + r;

                    // This is a match to a MapPoint in the map
                    if (vbMap[i])
                    {
                        cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0));
                        cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
                        mnTracked++;
                    }
                    else // This is match to a "visual odometry" MapPoint created in the last frame
                    {
                        cv::rectangle(im, pt1, pt2, cv::Scalar(255, 0, 0));
                        cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(255, 0, 0), -1);
                        mnTrackedVO++;
                    }
                }
            }
        }

        cv::Mat imWithInfo;
        DrawTextInfo(im, state, imWithInfo);

        return imWithInfo;
    }

    cv::Mat FrameDrawer::DrawFrameCurves()
    {
        static const cv::Scalar chainColors[] = {
            cv::Scalar(255, 80, 80),
            cv::Scalar(80, 255, 80),
            cv::Scalar(80, 80, 255),
            cv::Scalar(255, 220, 80),
            cv::Scalar(255, 80, 220),
            cv::Scalar(80, 220, 255),
            cv::Scalar(220, 160, 80),
            cv::Scalar(160, 80, 220)};
        const size_t colorCount = sizeof(chainColors) / sizeof(chainColors[0]);

        cv::Mat im;
        std::vector<BezierCurve> currentCurves;

        {
            std::unique_lock<std::mutex> lock(mMutex);
            im = mIm.clone();
            currentCurves = mpCurrentCurves;
        }

        if (im.channels() == 3)
        {
            cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
            im.convertTo(im, CV_8U, 1.0 / 1.3);
        }
        cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);

        std::vector<cv::Point> points;
        points.reserve(currentCurves.size());
        for (size_t i = 0; i < currentCurves.size(); ++i)
        {
            const std::vector<orderedEdgePoint> &samples = currentCurves[i].sampledPoints;
            for (size_t j = 0; j < samples.size(); ++j)
            {
                points.push_back(cv::Point(static_cast<int>(samples[j].x), static_cast<int>(samples[j].y)));
            }
            const size_t colorKey = currentCurves[i].hasEdgeChain() ? currentCurves[i].edgeChainId : i;
            cv::polylines(im, points, false, chainColors[colorKey % colorCount], 2, cv::LINE_AA);
            points.clear();
        }

        return im;
    }

    cv::Mat FrameDrawer::DrawCurveSampleCorrespondences()
    {
        cv::Mat image;
        std::vector<BezierCurve> observedCurves;
        std::vector<CurveSampleCorrespondence> correspondences;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            image = mIm.clone();
            observedCurves = mvCurveAssociationCurves;
            correspondences = mvCurveSampleCorrespondences;
        }

        if (image.empty())
            return cv::Mat();

        cv::Mat colorImage;
        if (image.channels() == 1)
            cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
        else if (image.channels() == 3)
            colorImage = image.clone();
        else if (image.channels() == 4)
            cv::cvtColor(image, colorImage, cv::COLOR_BGRA2BGR);
        else
            return cv::Mat();
        colorImage.convertTo(colorImage, CV_8UC3, 0.65);

        const cv::Scalar matchedBezierColor(80, 255, 80);
        const cv::Scalar projectedCurveColor(255, 80, 255);
        const cv::Scalar connectorColor(0, 255, 255);
        const cv::Scalar observedPointColor(255, 255, 80);

        cv::Mat projectedImage = colorImage.clone();
        cv::Mat observedImage = colorImage.clone();
        MapCurve *pDisplayedMapCurve = NULL;
        for (const CurveSampleCorrespondence &correspondence : correspondences)
        {
            if (!correspondence.pMapCurve ||
                !std::isfinite(correspondence.projectedPoint.x) ||
                !std::isfinite(correspondence.projectedPoint.y) ||
                !std::isfinite(correspondence.observedPoint.x) ||
                !std::isfinite(correspondence.observedPoint.y))
                continue;
            if (!pDisplayedMapCurve || correspondence.pMapCurve->mnId < pDisplayedMapCurve->mnId)
                pDisplayedMapCurve = correspondence.pMapCurve;
        }

        std::unordered_set<size_t> matchedObservedCurveIndices;
        std::vector<cv::Point2f> projectedSamples;
        size_t displayedCorrespondenceCount = 0;
        for (const CurveSampleCorrespondence &correspondence : correspondences)
        {
            if (correspondence.pMapCurve != pDisplayedMapCurve ||
                !std::isfinite(correspondence.projectedPoint.x) ||
                !std::isfinite(correspondence.projectedPoint.y) ||
                !std::isfinite(correspondence.observedPoint.x) ||
                !std::isfinite(correspondence.observedPoint.y))
                continue;

            matchedObservedCurveIndices.insert(correspondence.observedCurveIndex);
            projectedSamples.push_back(
                cv::Point2f(static_cast<float>(correspondence.projectedPoint.x),
                            static_cast<float>(correspondence.projectedPoint.y)));
            ++displayedCorrespondenceCount;
        }

        for (size_t curveIndex = 0; curveIndex < observedCurves.size(); ++curveIndex)
        {
            if (matchedObservedCurveIndices.count(curveIndex) == 0)
                continue;

            std::vector<cv::Point2f> samples;
            samples.reserve(observedCurves[curveIndex].sampledPoints.size());
            for (const orderedEdgePoint &sample : observedCurves[curveIndex].sampledPoints)
                samples.push_back(cv::Point2f(static_cast<float>(sample.x), static_cast<float>(sample.y)));

            DrawSampledCurve(observedImage, samples, matchedBezierColor, 2);
        }

        DrawSampledCurve(projectedImage, projectedSamples, projectedCurveColor, 2);

        cv::Mat correspondenceImage;
        cv::hconcat(projectedImage, observedImage, correspondenceImage);
        for (const CurveSampleCorrespondence &correspondence : correspondences)
        {
            if (correspondence.pMapCurve != pDisplayedMapCurve ||
                !std::isfinite(correspondence.projectedPoint.x) ||
                !std::isfinite(correspondence.projectedPoint.y) ||
                !std::isfinite(correspondence.observedPoint.x) ||
                !std::isfinite(correspondence.observedPoint.y))
                continue;

            const cv::Point2d observedPointInRightPanel(
                correspondence.observedPoint.x + projectedImage.cols,
                correspondence.observedPoint.y);
            DrawDashedLine(correspondenceImage, correspondence.projectedPoint,
                           observedPointInRightPanel, connectorColor, 1);

            const cv::Point projectedPoint(cvRound(correspondence.projectedPoint.x),
                                           cvRound(correspondence.projectedPoint.y));
            const cv::Point observedPoint(cvRound(observedPointInRightPanel.x),
                                          cvRound(correspondence.observedPoint.y));
            cv::circle(correspondenceImage, projectedPoint, 3, projectedCurveColor, -1, cv::LINE_AA);
            cv::drawMarker(correspondenceImage, observedPoint, observedPointColor,
                           cv::MARKER_CROSS, 7, 1, cv::LINE_AA);
        }

        cv::line(correspondenceImage,
                 cv::Point(projectedImage.cols, 0),
                 cv::Point(projectedImage.cols, correspondenceImage.rows - 1),
                 cv::Scalar(255, 255, 255), 1);

        const auto drawPanelTitle = [&correspondenceImage](const cv::Point &origin, const std::string &text)
        {
            cv::putText(correspondenceImage, text, origin,
                        cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(correspondenceImage, text, origin,
                        cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        };
        const std::string mapCurveTitle = pDisplayedMapCurve
                                              ? "MapCurve " + std::to_string(pDisplayedMapCurve->mnId) + " projection"
                                              : "No matched MapCurve";
        drawPanelTitle(cv::Point(12, 28), mapCurveTitle);
        drawPanelTitle(cv::Point(projectedImage.cols + 12, 28), "Current Bezier curves");

        const std::string countText = std::to_string(displayedCorrespondenceCount) + " fixed samples";
        cv::putText(correspondenceImage, countText, cv::Point(12, correspondenceImage.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(correspondenceImage, countText, cv::Point(12, correspondenceImage.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        return correspondenceImage;
    }

    cv::Scalar FrameDrawer::AssociationColor(const size_t colorIndex)
    {
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(static_cast<int>((colorIndex * 37) % 180), 230, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        const cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
        return cv::Scalar(color[0], color[1], color[2]);
    }

    void FrameDrawer::DrawSampledCurve(cv::Mat &image, const std::vector<cv::Point2f> &samples, const cv::Scalar &color, const int thickness)
    {
        if (samples.empty())
            return;
        if (samples.size() == 1)
        {
            cv::circle(image, cv::Point(cvRound(samples[0].x), cvRound(samples[0].y)), std::max(1, thickness), color, -1, cv::LINE_AA);
            return;
        }
        for (size_t sampleIndex = 1; sampleIndex < samples.size(); ++sampleIndex)
        {
            const cv::Point first(cvRound(samples[sampleIndex - 1].x), cvRound(samples[sampleIndex - 1].y));
            const cv::Point second(cvRound(samples[sampleIndex].x), cvRound(samples[sampleIndex].y));
            cv::line(image, first, second, color, thickness, cv::LINE_AA);
        }
    }

    bool FrameDrawer::ProjectWorldPoint(const cv::Point3d &worldPoint, const cv::Mat &Tcw, cv::Point2f &imagePoint)
    {
        if (Tcw.empty())
            return false;
        const float worldX = static_cast<float>(worldPoint.x);
        const float worldY = static_cast<float>(worldPoint.y);
        const float worldZ = static_cast<float>(worldPoint.z);
        const float cameraX = Tcw.at<float>(0, 0) * worldX + Tcw.at<float>(0, 1) * worldY + Tcw.at<float>(0, 2) * worldZ + Tcw.at<float>(0, 3);
        const float cameraY = Tcw.at<float>(1, 0) * worldX + Tcw.at<float>(1, 1) * worldY + Tcw.at<float>(1, 2) * worldZ + Tcw.at<float>(1, 3);
        const float cameraZ = Tcw.at<float>(2, 0) * worldX + Tcw.at<float>(2, 1) * worldY + Tcw.at<float>(2, 2) * worldZ + Tcw.at<float>(2, 3);
        if (!std::isfinite(cameraX) || !std::isfinite(cameraY) || !std::isfinite(cameraZ) || cameraZ <= 0.0f)
            return false;
        imagePoint.x = Frame::fx * cameraX / cameraZ + Frame::cx;
        imagePoint.y = Frame::fy * cameraY / cameraZ + Frame::cy;
        return std::isfinite(imagePoint.x) && std::isfinite(imagePoint.y);
    }

    void FrameDrawer::AppendResampledSegment(const cv::Point2f &first, const cv::Point2f &second, std::vector<cv::Point2f> &samples)
    {
        const float length = cv::norm(second - first);
        if (length <= 1e-3f)
            return;
        const int intervalCount = std::max(1, static_cast<int>(std::ceil(length / 3.0f)));
        for (int interval = 1; interval <= intervalCount; ++interval)
        {
            const float ratio = static_cast<float>(interval) / intervalCount;
            samples.push_back(first + ratio * (second - first));
        }
    }

    bool FrameDrawer::ProjectMapCurve(MapCurve *pMapCurve, const cv::Mat &Tcw, const float margin, std::vector<cv::Point2f> &projectedSamples)
    {
        projectedSamples.clear();
        if (!pMapCurve || pMapCurve->isBad() || Tcw.empty())
            return false;
        const std::vector<cv::Point3d> worldPoints = pMapCurve->GetCurvePoints();
        bool hasPreviousPoint = false;
        cv::Point2f previousPoint;
        for (const cv::Point3d &worldPoint : worldPoints)
        {
            cv::Point2f currentPoint;
            if (!ProjectWorldPoint(worldPoint, Tcw, currentPoint) || currentPoint.x < Frame::mnMinX - margin || currentPoint.x > Frame::mnMaxX + margin || currentPoint.y < Frame::mnMinY - margin || currentPoint.y > Frame::mnMaxY + margin)
            {
                hasPreviousPoint = false;
                continue;
            }
            if (!hasPreviousPoint)
                projectedSamples.push_back(currentPoint);
            else
                AppendResampledSegment(previousPoint, currentPoint, projectedSamples);
            previousPoint = currentPoint;
            hasPreviousPoint = true;
        }
        return projectedSamples.size() >= 2;
    }

    cv::Mat FrameDrawer::DrawCurveAssociations()
    {
        cv::Mat image;
        cv::Mat Tcw;
        std::vector<BezierCurve> observedCurves;
        std::vector<MapCurve *> mapCurveMatches;
        std::vector<CurveMatchDiagnostic> matchDiagnostics;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            image = mIm.clone();
            Tcw = mCurveAssociationTcw.clone();
            observedCurves = mvCurveAssociationCurves;
            mapCurveMatches = mvpCurveAssociationMatches;
            matchDiagnostics = mvCurveMatchDiagnostics;
        }
        if (image.empty() || Tcw.empty() || matchDiagnostics.empty())
            return cv::Mat();

        cv::Mat colorImage;
        if (image.channels() == 1)
            cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
        else if (image.channels() == 3)
            colorImage = image.clone();
        else if (image.channels() == 4)
            cv::cvtColor(image, colorImage, cv::COLOR_BGRA2BGR);
        else
            return cv::Mat();
        colorImage.convertTo(colorImage, CV_8UC3, 0.65);

        cv::Mat projectedImage = colorImage.clone();
        cv::Mat observedImage = colorImage.clone();
        cv::Mat unmatchedImage = colorImage.clone();
        cv::Mat diagnosticImage(colorImage.size(), CV_8UC3, cv::Scalar(24, 24, 24));
        std::unordered_map<MapCurve *, cv::Scalar> associatedColors;
        const size_t associationCount = std::min(observedCurves.size(), mapCurveMatches.size());
        for (size_t curveIndex = 0; curveIndex < associationCount; ++curveIndex)
        {
            MapCurve *pMapCurve = mapCurveMatches[curveIndex];
            if (pMapCurve && associatedColors.count(pMapCurve) == 0)
            {
                const cv::Scalar color = AssociationColor(pMapCurve->mnId);
                associatedColors[pMapCurve] = color;
            }
        }

        struct CurveTextLabel
        {
            cv::Point2f position;
            std::string text;
            cv::Scalar color;
            double fontScale;
        };

        const auto curveMidpoint = [](const std::vector<cv::Point2f> &samples)
        {
            if (samples.size() == 1)
                return samples.front();

            float totalLength = 0.0f;
            for (size_t sampleIndex = 1; sampleIndex < samples.size(); ++sampleIndex)
                totalLength += cv::norm(samples[sampleIndex] - samples[sampleIndex - 1]);

            if (totalLength <= 1e-3f)
                return samples[samples.size() / 2];

            const float targetLength = totalLength * 0.5f;
            float accumulatedLength = 0.0f;
            for (size_t sampleIndex = 1; sampleIndex < samples.size(); ++sampleIndex)
            {
                const cv::Point2f segment = samples[sampleIndex] - samples[sampleIndex - 1];
                const float segmentLength = cv::norm(segment);
                if (accumulatedLength + segmentLength >= targetLength)
                {
                    const float ratio = segmentLength > 1e-6f ? (targetLength - accumulatedLength) / segmentLength : 0.0f;
                    return samples[sampleIndex - 1] + ratio * segment;
                }
                accumulatedLength += segmentLength;
            }
            return samples.back();
        };

        const auto drawCurveLabels = [](cv::Mat &targetImage, const std::vector<CurveTextLabel> &labels)
        {
            for (const CurveTextLabel &label : labels)
            {
                int baseline = 0;
                const cv::Size textSize = cv::getTextSize(label.text, cv::FONT_HERSHEY_SIMPLEX, label.fontScale, 1, &baseline);
                cv::Point origin(cvRound(label.position.x - 0.5f * textSize.width),
                                 cvRound(label.position.y + 0.5f * textSize.height));
                origin.x = std::max(2, std::min(origin.x, std::max(2, targetImage.cols - textSize.width - 2)));
                origin.y = std::max(textSize.height + 2, std::min(origin.y, std::max(textSize.height + 2, targetImage.rows - baseline - 2)));

                // Draw a curve-colored outline first, then place the white ID on top.
                cv::putText(targetImage, label.text, origin, cv::FONT_HERSHEY_SIMPLEX, label.fontScale, label.color, 3, cv::LINE_AA);
                cv::putText(targetImage, label.text, origin, cv::FONT_HERSHEY_SIMPLEX, label.fontScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        };

        const auto drawPanelTitle = [](cv::Mat &targetImage, const std::string &title)
        {
            cv::putText(targetImage, title, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(targetImage, title, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        };

        std::vector<CurveTextLabel> projectedLabels;
        std::vector<CurveTextLabel> observedLabels;
        std::vector<CurveTextLabel> unmatchedLabels;
        std::unordered_set<MapCurve *> drawnProjectedMapCurves;
        for (MapCurve *pMapCurve : mapCurveMatches)
        {
            if (!pMapCurve || !drawnProjectedMapCurves.insert(pMapCurve).second)
                continue;

            std::vector<cv::Point2f> projectedSamples;
            if (!ProjectMapCurve(pMapCurve, Tcw, 0.0f, projectedSamples))
                continue;
            const cv::Scalar color = AssociationColor(pMapCurve->mnId);
            DrawSampledCurve(projectedImage, projectedSamples, color, 2);
            projectedLabels.push_back(CurveTextLabel{curveMidpoint(projectedSamples), std::to_string(pMapCurve->mnId), color, 0.5});
        }

        std::unordered_set<MapCurve *> diagnosedMapCurves;
        std::vector<const CurveMatchDiagnostic *> unmatchedDiagnostics;
        for (const CurveMatchDiagnostic &diagnostic : matchDiagnostics)
        {
            MapCurve *pMapCurve = diagnostic.pMapCurve;
            if (!pMapCurve || !diagnosedMapCurves.insert(pMapCurve).second)
                continue;

            if (!diagnostic.matched)
                unmatchedDiagnostics.push_back(&diagnostic);
        }

        for (const CurveMatchDiagnostic *pDiagnostic : unmatchedDiagnostics)
        {
            MapCurve *pMapCurve = pDiagnostic->pMapCurve;
            std::vector<cv::Point2f> projectedSamples;
            if (!ProjectMapCurve(pMapCurve, Tcw, 0.0f, projectedSamples))
                continue;
            const cv::Scalar color = AssociationColor(pMapCurve->mnId);
            DrawSampledCurve(unmatchedImage, projectedSamples, color, 2);
            const std::string label = std::to_string(pMapCurve->mnId) + " " + pDiagnostic->failureType;
            unmatchedLabels.push_back(CurveTextLabel{curveMidpoint(projectedSamples), label, color, 0.42});
        }

        for (size_t curveIndex = 0; curveIndex < observedCurves.size(); ++curveIndex)
        {
            std::vector<cv::Point2f> observedSamples;
            observedSamples.reserve(observedCurves[curveIndex].sampledPoints.size());
            for (const orderedEdgePoint &sample : observedCurves[curveIndex].sampledPoints)
                observedSamples.push_back(cv::Point2f(static_cast<float>(sample.x), static_cast<float>(sample.y)));
            MapCurve *pMapCurve = curveIndex < mapCurveMatches.size() ? mapCurveMatches[curveIndex] : NULL;
            const auto color = associatedColors.find(pMapCurve);
            DrawSampledCurve(observedImage, observedSamples, color == associatedColors.end() ? cv::Scalar(110, 110, 110) : color->second, 2);
            if (pMapCurve && color != associatedColors.end() && !observedSamples.empty())
                observedLabels.push_back(CurveTextLabel{curveMidpoint(observedSamples), std::to_string(pMapCurve->mnId), color->second, 0.5});
        }

        drawCurveLabels(projectedImage, projectedLabels);
        drawCurveLabels(observedImage, observedLabels);
        drawCurveLabels(unmatchedImage, unmatchedLabels);

        const int diagnosticTop = 48;
        const int diagnosticRowHeight = 18;
        const int maximumRows = std::max(1, (diagnosticImage.rows - diagnosticTop - 8) / diagnosticRowHeight);
        const int diagnosticColumns = std::max(1, static_cast<int>((unmatchedDiagnostics.size() + maximumRows - 1) / maximumRows));
        const int diagnosticColumnWidth = std::max(1, diagnosticImage.cols / diagnosticColumns);
        for (size_t diagnosticIndex = 0; diagnosticIndex < unmatchedDiagnostics.size(); ++diagnosticIndex)
        {
            const CurveMatchDiagnostic &diagnostic = *unmatchedDiagnostics[diagnosticIndex];
            const int column = static_cast<int>(diagnosticIndex) / maximumRows;
            const int row = static_cast<int>(diagnosticIndex) % maximumRows;
            const int x = column * diagnosticColumnWidth + 8;
            const int y = diagnosticTop + row * diagnosticRowHeight;
            const std::string text = "ID " + std::to_string(diagnostic.pMapCurve->mnId) + ": " + diagnostic.failureDetail;
            int baseline = 0;
            const cv::Size nominalSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
            const double availableWidth = std::max(12, diagnosticColumnWidth - 16);
            const double fontScale = nominalSize.width > availableWidth
                                         ? std::max(0.22, 0.42 * availableWidth / nominalSize.width)
                                         : 0.42;
            const cv::Scalar color = AssociationColor(diagnostic.pMapCurve->mnId);
            cv::rectangle(diagnosticImage, cv::Point(x, y - 8), cv::Point(x + 5, y - 3), color, -1, cv::LINE_AA);
            cv::putText(diagnosticImage, text, cv::Point(x + 10, y), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        }

        drawPanelTitle(projectedImage, "Matched projected MapCurves");
        drawPanelTitle(observedImage, "Current Bezier associations");
        drawPanelTitle(unmatchedImage, "Unmatched projected MapCurves");
        drawPanelTitle(diagnosticImage, "Unmatched MapCurve diagnostics");

        cv::Mat topRow;
        cv::Mat bottomRow;
        cv::Mat associationImage;
        cv::hconcat(projectedImage, observedImage, topRow);
        cv::hconcat(unmatchedImage, diagnosticImage, bottomRow);
        cv::vconcat(topRow, bottomRow, associationImage);
        cv::line(associationImage, cv::Point(projectedImage.cols, 0), cv::Point(projectedImage.cols, associationImage.rows - 1), cv::Scalar(255, 255, 255), 1);
        cv::line(associationImage, cv::Point(0, projectedImage.rows), cv::Point(associationImage.cols - 1, projectedImage.rows), cv::Scalar(255, 255, 255), 1);
        return associationImage;
    }

    void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
    {
        stringstream s;
        if (nState == Tracking::NO_IMAGES_YET)
            s << " WAITING FOR IMAGES";
        else if (nState == Tracking::NOT_INITIALIZED)
            s << " TRYING TO INITIALIZE ";
        else if (nState == Tracking::OK)
        {
            if (!mbOnlyTracking)
                s << "SLAM MODE |  ";
            else
                s << "LOCALIZATION | ";
            int nKFs = mpMap->KeyFramesInMap();
            int nMPs = mpMap->MapPointsInMap();
            s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
            if (mnTrackedVO > 0)
                s << ", + VO matches: " << mnTrackedVO;
        }
        else if (nState == Tracking::LOST)
        {
            s << " TRACK LOST. TRYING TO RELOCALIZE ";
        }
        else if (nState == Tracking::SYSTEM_NOT_READY)
        {
            s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
        im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
        imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
        cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
    }

    void FrameDrawer::Update(Tracking *pTracker)
    {
        unique_lock<mutex> lock(mMutex);
        pTracker->mImGray.copyTo(mIm);
        mCurveAssociationTcw = pTracker->mCurrentFrame.mTcw.clone();
        mvCurveAssociationCurves = pTracker->mCurrentFrame.mvBezierCurves;
        mvpCurveAssociationMatches = pTracker->mCurrentFrame.mvpMapCurves;
        mvpCurveAssociationCandidates = pTracker->mvpCurveAssociationCandidates;
        mvCurveMatchDiagnostics = pTracker->mCurrentFrame.mvCurveMatchDiagnostics;
        mvCurveSampleCorrespondences = pTracker->mCurrentFrame.mvCurveSampleCorrespondences;
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
        mpCurrentCurves = pTracker->mCurrentFrame.mvBezierCurves;
        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N, false);
        mvbMap = vector<bool>(N, false);
        mbOnlyTracking = pTracker->mbOnlyTracking;

        if (pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED)
        {
            mvIniKeys = pTracker->mInitialFrame.mvKeys;
            mvIniMatches = pTracker->mvIniMatches;
        }
        else if (pTracker->mLastProcessedState == Tracking::OK)
        {
            for (int i = 0; i < N; i++)
            {
                MapPoint *pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    if (!pTracker->mCurrentFrame.mvbOutlier[i])
                    {
                        if (pMP->Observations() > 0)
                            mvbMap[i] = true;
                        else
                            mvbVO[i] = true;
                    }
                }
            }
        }
        mState = static_cast<int>(pTracker->mLastProcessedState);
    }

} // namespace ORB_SLAM
