#include "Curve/CurveConfig.h"

#include <opencv2/core/core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ORB_SLAM2
{

    int ReadInt(const cv::FileStorage &settings, const char *name, int defaultValue)
    {
        const cv::FileNode node = settings[name];
        return node.empty() ? defaultValue : static_cast<int>(node);
    }

    float ReadFloat(const cv::FileStorage &settings, const char *name, float defaultValue)
    {
        const cv::FileNode node = settings[name];
        return node.empty() ? defaultValue : static_cast<float>(node);
    }

    CurveConfig::CurveConfig(const std::string &settingsPath, bool isRGBD)
    {
        cv::FileStorage settings(settingsPath, cv::FileStorage::READ);
        if (!settings.isOpened())
            throw std::runtime_error("Failed to open settings file: " + settingsPath);

        enabled = ReadInt(settings, "Curve.UseCurve", 0) != 0 && isRGBD;
        minDepth = ReadFloat(settings, "Curve.MinDepth", 0.02f);
        maxDepth = ReadFloat(settings, "Curve.MaxDepth", 4.0f);
        BezierFitter = ReadInt(settings, "Curve.BezierFitter", 2);
        depthMedianWindow = std::max(1, ReadInt(settings, "Curve.DepthMedianWindow", 5));
        if (depthMedianWindow % 2 == 0)
            ++depthMedianWindow;
        maxNeighborDepthJump = std::max(0.0f, ReadFloat(settings, "Curve.MaxNeighborDepthJump", 0.08f));
        maxNeighborDepthJumpRatio = std::max(0.0f, ReadFloat(settings, "Curve.MaxNeighborDepthJumpRatio", 0.03f));
        minimumDepthSegmentSamples = std::max(
            4,
            ReadInt(
                settings,
                "Curve.MinimumDepthSegmentSamples",
                4));
        minimumExtensionSamples = std::max(2, ReadInt(settings, "Curve.MinimumExtensionSamples", 3));
        fusionMaxPixelDistance = std::max(0.0f, ReadFloat(settings, "Curve.FusionMaxPixelDistance", 6.0f));
        fusionMax3DDistance = std::max(0.0f, ReadFloat(settings, "Curve.FusionMax3DDistance", 0.10f));
        extensionMaxPixelGap = std::max(0.0f, ReadFloat(settings, "Curve.ExtensionMaxPixelGap", 10.0f));
        extensionMax3DGap = std::max(0.0f, ReadFloat(settings, "Curve.ExtensionMax3DGap", 0.15f));
        depthSegmentMax3DGap = std::min(
            extensionMax3DGap,
            std::max(
                0.0f,
                ReadFloat(
                    settings,
                    "Curve.DepthSegmentMax3DGap",
                    extensionMax3DGap)));
        minimumJoinDirectionCosine = std::max(-1.0f, std::min(1.0f, ReadFloat(settings, "Curve.MinimumJoinDirectionCosine", 0.5f)));
        geometrySmoothingWeight = std::max(0.0f, std::min(0.5f, ReadFloat(settings, "Curve.GeometrySmoothingWeight", 0.15f)));
        geometryQualityReferenceSpacing = std::max(0.001f, ReadFloat(settings, "Curve.GeometryQualityReferenceSpacing", 0.02f));
        geometryQualityHighTurnDegrees = std::max(0.0f, std::min(180.0f, ReadFloat(settings, "Curve.GeometryQualityHighTurnDegrees", 30.0f)));
        geometryQualityMinimumBendDegrees = std::max(0.0f, std::min(geometryQualityHighTurnDegrees, ReadFloat(settings, "Curve.GeometryQualityMinimumBendDegrees", 10.0f)));
        geometryQualityHardUTurnDegrees = std::max(geometryQualityHighTurnDegrees, std::min(180.0f, ReadFloat(settings, "Curve.GeometryQualityHardUTurnDegrees", 150.0f)));
        geometryQualityMaximumHighTurnRatio = std::max(0.0f, std::min(1.0f, ReadFloat(settings, "Curve.GeometryQualityMaximumHighTurnRatio", 0.25f)));
        geometryQualityMaximumBendFlipRatio = std::max(0.0f, std::min(1.0f, ReadFloat(settings, "Curve.GeometryQualityMaximumBendFlipRatio", 0.25f)));
        geometryQualityBendFlipCosine = std::max(-1.0f, std::min(1.0f, ReadFloat(settings, "Curve.GeometryQualityBendFlipCosine", -0.3f)));
        geometryQualityMaximumBendSeparation = std::max(0.0f, ReadFloat(settings, "Curve.GeometryQualityMaximumBendSeparation", 0.15f));
        geometryQualityMinimumHighTurns = std::max(1, ReadInt(settings, "Curve.GeometryQualityMinimumHighTurns", 2));
        geometryQualityMinimumBendFlips = std::max(1, ReadInt(settings, "Curve.GeometryQualityMinimumBendFlips", 2));
        geometryQualityMinimumHardUTurns = std::max(1, ReadInt(settings, "Curve.GeometryQualityMinimumHardUTurns", 2));
        geometryQualityCullingMinAge = std::max(0, ReadInt(settings, "Curve.GeometryQualityCullingMinAge", 2));
        maximumFusionWeight = std::max(1, ReadInt(settings, "Curve.MaximumFusionWeight", 10));
        mapFusionMinimum3DCoverage = std::max(0.0f, std::min(1.0f, ReadFloat(settings, "Curve.MapFusionMinimum3DCoverage", 0.5f)));
        mapFusionMaximumP90Distance = std::max(0.0f, ReadFloat(settings, "Curve.MapFusionMaximumP90Distance", 0.15f));
        matchSearchRadius = ReadFloat(settings, "Curve.MatchSearchRadius", 6.0f);
        minCandidateHits = ReadInt(settings, "Curve.MinCandidateHits", 3);
        minCandidateCoverage = ReadFloat(settings, "Curve.MinCandidateCoverage", 0.2f);
        unmatchedCost = ReadFloat(settings, "Curve.UnmatchedCost", 6.0f);
    }

} // namespace ORB_SLAM2
