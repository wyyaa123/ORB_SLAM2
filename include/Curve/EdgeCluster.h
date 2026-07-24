#ifndef EDGECLUSTER_H
#define EDGECLUSTER_H

#include <opencv2/core/core.hpp>
#include <vector>

namespace ORB_SLAM2
{

    class orderedEdgePoint
    {
    public:
        // ###########################
        //--      2D相关成员
        // ###########################

        //-- 像素坐标
        double x = 0.0;
        double y = 0.0;
        //-- RGBD给的深度
        float depth = 0.0f;
        // ###########################
        //--      3D相关成员
        // ###########################
        double x_3d = 0.0;
        double y_3d = 0.0;
        double z_3d = 0.0;
        orderedEdgePoint(double _x, double _y)
        {
            x = _x;
            y = _y;
        }
    };

    /**
     * @brief 无序的边缘点，仅在边缘组织的过程中使用到，
     * @details 存储度、当前ID与父节点ID等信息，通过拓展的拓扑关系将canny散点组织成有序边缘
     */
    struct edgePoint
    {
        //-- 边缘点的像素坐标
        cv::Point pixel;
        //-- 边缘点的父节点的ID
        int father_id;
        //-- 边缘点的入度（该节点是几个点的父节点）
        int degree;
        //-- 是否是区域生长的遍历起点（root）
        bool isRoot;

        edgePoint(cv::Point _pixel, int _father_id)
        {
            pixel = _pixel;
            father_id = _father_id;
            degree = 0;
            isRoot = false;
        }
    };

    /**
     * @brief 无序的边缘聚类，存储一组无序但可以自组织的edgePoint，可实现edgePoint的自组织
     * @details 存储edgePoint, 利用edgePoint中的拓扑关系进行边缘的自组织
     */
    class EdgeCluster
    {
    public:
        //-- 每个边缘的点列表，可以用push_back进行操作
        std::vector<edgePoint> mvPoints;
        EdgeCluster() {}

        EdgeCluster(std::vector<edgePoint> list);

        //-- 整理边缘，根据区域生长建立的邻接关系将边缘整理成有序的
        std::vector<edgePoint> organize();

    private:
        //-- 计算每个点的入度（即该节点是几个点的父节点）
        void calculateDegree();
    };

    using Edge = std::vector<orderedEdgePoint>;
}

#endif
