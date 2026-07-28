#ifndef CURVEMATCHER_H
#define CURVEMATCHER_H

#include <cstddef>
#include <unordered_set>
#include <vector>

#include <opencv2/core/core.hpp>

#include "Curve/CurveConfig.h"

namespace ORB_SLAM2
{
    class Frame;
    class MapCurve;
    class orderedEdgePoint;

    struct CurveSampleRef
    {
        CurveSampleRef(std::size_t curveIndex, std::size_t sampleIndex);

        std::size_t curveIndex;
        std::size_t sampleIndex;
    };

    struct ProjectedMapCurve
    {
        ProjectedMapCurve();

        MapCurve *pMapCurve;
        std::vector<cv::Point2f> samples;
    };

    struct CandidateStatistics
    {
        CandidateStatistics();

        void AddHit(std::size_t observedSampleIndex);

        std::size_t matchedMapSamples;
        std::unordered_set<std::size_t> matchedObservedSamples;
    };

    struct CurveSimilarity
    {
        CurveSimilarity();

        double cost;
        double meanDistance;
        double meanDirectionCost;
        double dtwCost;
        int minMapIndex;
        int maxMapIndex;
        bool valid;
    };

    struct AcceptedFragment
    {
        AcceptedFragment(std::size_t curveIndex, const CurveSimilarity &similarity);

        std::size_t curveIndex;
        CurveSimilarity similarity;
    };

    class CurveSampleGrid
    {
    public:
        CurveSampleGrid(float minX, float minY, float maxX, float maxY, float cellSize);

        void Insert(const cv::Point2f &point, const CurveSampleRef &sample);
        void Query(const cv::Point2f &center, float radius, std::vector<CurveSampleRef> &samples) const;

    private:
        int ColumnFor(float x) const;
        int RowFor(float y) const;
        bool IsValidCell(int column, int row) const;
        std::size_t CellIndex(int column, int row) const;

        float mMinX;
        float mMinY;
        float mCellSize;
        int mColumns;
        int mRows;
        std::vector<std::vector<CurveSampleRef>> mCells;
    };

    struct DirectedCurveMetrics
    {
        DirectedCurveMetrics();

        double meanDistance;
        double meanDirectionCost;
        int minReferenceIndex;
        int maxReferenceIndex;
    };

    class CurveMatcher
    {
    public:
        explicit CurveMatcher(const CurveConfigPtr &curveConfig);

        // Associate current-frame observations with an explicit map-curve
        // set from a reference keyframe, the last frame, or the local map.
        int AssociateMapCurvesToFrame(const std::vector<MapCurve *> &mapCurves, Frame &currentFrame);

    protected:
        CurveConfigPtr mCurveConfig;

    private:
        cv::Point2f ToPoint2f(const orderedEdgePoint &point);
        bool ProjectWorldPoint(const cv::Point3d &worldPoint, const Frame &frame, cv::Point2f &imagePoint);
        void AppendResampledSegment(const cv::Point2f &first, const cv::Point2f &second, std::vector<cv::Point2f> &samples);
        bool ProjectMapCurve(MapCurve *pMapCurve, const Frame &frame, float margin, ProjectedMapCurve &projected);
        std::vector<cv::Point2f> LimitSamples(const std::vector<cv::Point2f> &points, std::size_t maximumSamples);
        cv::Point2f CurveTangentAt(const std::vector<cv::Point2f> &points, std::size_t index);
        double LocalAlignmentCost(const std::vector<cv::Point2f> &reference, std::size_t referenceIndex, const std::vector<cv::Point2f> &query, std::size_t queryIndex);
        double SubsequenceDtwCost(const std::vector<cv::Point2f> &reference, const std::vector<cv::Point2f> &query);
        double OrientationIndependentDtw(const std::vector<cv::Point2f> &reference, const std::vector<cv::Point2f> &query);
        DirectedCurveMetrics ComputeDirectedMetrics(const std::vector<cv::Point2f> &reference, const std::vector<cv::Point2f> &query);
        CurveSimilarity ComputeCurveSimilarity(const ProjectedMapCurve &mapCurve, const std::vector<cv::Point2f> &observedCurve, float maximumMeanDistance);
        bool EndpointsAreContinuous(const std::vector<cv::Point2f> &first, const std::vector<cv::Point2f> &second);
        bool FragmentIsCompatible(std::size_t candidateCurveIndex, const CurveSimilarity &candidateSimilarity, const std::vector<AcceptedFragment> &acceptedFragments, const Frame &frame, const std::vector<std::vector<cv::Point2f>> &observedCurves, std::size_t projectedMapSampleCount);
        void BuildFixedSampleCorrespondences(Frame &frame);
    };
}

#endif // CURVEMATCHER_H
