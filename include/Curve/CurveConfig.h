#ifndef CURVE_CONFIG_H
#define CURVE_CONFIG_H

#include <memory>
#include <string>

namespace ORB_SLAM2
{

    class CurveConfig;
    using CurveConfigPtr = std::shared_ptr<const CurveConfig>;

    class CurveConfig
    {
    public:
        CurveConfig(const std::string &settingsPath, bool isRGBD);

        bool enabled;
        float minDepth;
        float maxDepth;
        float validRatio;
        int BezierFitter;

        // Curve depth denoising parameters.
        int depthMedianWindow;
        float maxNeighborDepthJump;
        float maxNeighborDepthJumpRatio;
        int minimumExtensionSamples;
        float fusionMaxPixelDistance;
        float fusionMax3DDistance;
        float extensionMaxPixelGap;
        float extensionMax3DGap;
        float minimumJoinDirectionCosine;
        float geometrySmoothingWeight;
        int maximumFusionWeight;
        float mapFusionMinimum3DCoverage;
        float mapFusionMaximumP90Distance;

        // Curve association parameters.
        float matchSearchRadius;
        int minCandidateHits;
        float minCandidateCoverage;
        float unmatchedCost;
    };

} // namespace ORB_SLAM2

#endif // CURVE_CONFIG_H
