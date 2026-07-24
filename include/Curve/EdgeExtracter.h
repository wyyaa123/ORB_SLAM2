#ifndef EDGE_EXTRACTER_H
#define EDGE_EXTRACTER_H

#include "EdgeCluster.h"

namespace ORB_SLAM2
{

    class EdgeExtracter
    {
    public:
        EdgeExtracter(double _angleThreshold, double _threshold1, double _threshold2) : angleThreshold(_angleThreshold), threshold1(_threshold1), threshold2(_threshold2) {}

        std::vector<Edge> operator()(const cv::Mat &imGray, const cv::Mat &imSem = cv::Mat());

    private:
        inline float calcAngleBias(float angle_1, float angle_2);

        void preprocessSem(const cv::Mat &imSem);

        void preprocessEdge();

        void cvt2OrderedEdges();

        void regionGrowthClusteringOCanny(float angle_Thres = 20.0f, const cv::Point &offset = cv::Point(0, 0));

        cv::Mat mSem;
        cv::Mat mMatGradAngle;
        cv::Mat mCanny;
        double angleThreshold;
        double threshold1;
        double threshold2;

        std::vector<Edge> mvEdges;
        std::vector<EdgeCluster> mvEdgeClusters;
    };
}

#endif
