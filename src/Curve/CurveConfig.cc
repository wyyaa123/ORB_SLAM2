#include "Curve/CurveConfig.h"

#include <opencv2/core/core.hpp>

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

        enabled = ReadInt(settings, "Curve.UseCurve", 0) != 0;
        minDepth = ReadFloat(settings, "Curve.MinDepth", 0.02f);
        maxDepth = ReadFloat(settings, "Curve.MaxDepth", 4.0f);
        validRatio = ReadFloat(settings, "Curve.ValidRatio", 0.3f);
    }

} // namespace ORB_SLAM2
