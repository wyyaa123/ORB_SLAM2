#include "Curve/CurveProjectionEdge.h"

#include <cmath>

namespace ORB_SLAM2
{
    EdgeSE3ProjectCurveOnlyPose::EdgeSE3ProjectCurveOnlyPose() : fx(0.0), fy(0.0), cx(0.0), cy(0.0)
    {
        observedPoint.setZero();
        observedNormal.setZero();
        Xw.setZero();
    }

    void EdgeSE3ProjectCurveOnlyPose::computeError()
    {
        const g2o::VertexSE3Expmap *vertex = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const Eigen::Vector2d projectedPoint = Project(vertex->estimate().map(Xw));
        _error[0] = observedNormal.dot(observedPoint - projectedPoint);
    }

    void EdgeSE3ProjectCurveOnlyPose::linearizeOplus()
    {
        const g2o::VertexSE3Expmap *vertex = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const Eigen::Vector3d cameraPoint = vertex->estimate().map(Xw);
        const double x = cameraPoint[0];
        const double y = cameraPoint[1];
        const double inverseZ = 1.0 / cameraPoint[2];
        const double inverseZSquared = inverseZ * inverseZ;

        Eigen::Matrix<double, 2, 6> projectionJacobian;
        projectionJacobian(0, 0) = x * y * inverseZSquared * fx;
        projectionJacobian(0, 1) = -(1.0 + x * x * inverseZSquared) * fx;
        projectionJacobian(0, 2) = y * inverseZ * fx;
        projectionJacobian(0, 3) = -inverseZ * fx;
        projectionJacobian(0, 4) = 0.0;
        projectionJacobian(0, 5) = x * inverseZSquared * fx;
        projectionJacobian(1, 0) = (1.0 + y * y * inverseZSquared) * fy;
        projectionJacobian(1, 1) = -x * y * inverseZSquared * fy;
        projectionJacobian(1, 2) = -x * inverseZ * fy;
        projectionJacobian(1, 3) = 0.0;
        projectionJacobian(1, 4) = -inverseZ * fy;
        projectionJacobian(1, 5) = y * inverseZSquared * fy;
        _jacobianOplusXi = observedNormal.transpose() * projectionJacobian;
    }

    bool EdgeSE3ProjectCurveOnlyPose::isDepthPositive() const
    {
        const g2o::VertexSE3Expmap *vertex = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        return vertex->estimate().map(Xw)[2] > 0.0;
    }

    bool EdgeSE3ProjectCurveOnlyPose::read(std::istream &input)
    {
        return false;
    }

    bool EdgeSE3ProjectCurveOnlyPose::write(std::ostream &output) const
    {
        return false;
    }

    Eigen::Vector2d EdgeSE3ProjectCurveOnlyPose::Project(const Eigen::Vector3d &cameraPoint) const
    {
        const double inverseZ = 1.0 / cameraPoint[2];
        return Eigen::Vector2d(fx * cameraPoint[0] * inverseZ + cx, fy * cameraPoint[1] * inverseZ + cy);
    }
}
