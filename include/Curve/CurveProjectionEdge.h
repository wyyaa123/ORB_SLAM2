#ifndef CURVEPROJECTIONEDGE_H
#define CURVEPROJECTIONEDGE_H

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace ORB_SLAM2
{
    class EdgeSE3ProjectCurveOnlyPose : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeSE3ProjectCurveOnlyPose();

        void computeError();
        void linearizeOplus();
        bool isDepthPositive() const;
        bool read(std::istream &input);
        bool write(std::ostream &output) const;

        Eigen::Vector2d Project(const Eigen::Vector3d &cameraPoint) const;

        Eigen::Vector3d Xw;
        Eigen::Vector2d observedPoint;
        Eigen::Vector2d observedNormal;
        double fx;
        double fy;
        double cx;
        double cy;
    };
}

#endif // CURVEPROJECTIONEDGE_H
