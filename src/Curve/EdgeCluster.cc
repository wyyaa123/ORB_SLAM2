#include "Curve/EdgeCluster.h"

#include <algorithm>
#include <utility>

namespace ORB_SLAM2
{

    EdgeCluster::EdgeCluster(std::vector<edgePoint> list)
        : mvPoints(std::move(list))
    {
    }

    std::vector<edgePoint> EdgeCluster::organize()
    {
        //-- 确保每个节点的度都是正确的
        calculateDegree();

        //-- 收集所有末端节点（没有入度 degree=0）
        std::vector<size_t> endNodes;
        endNodes.reserve(mvPoints.size() / 2); //-- 预分配空间
        for (size_t i = 0; i < mvPoints.size(); ++i)
        {
            if (mvPoints[i].degree == 0)
                endNodes.push_back(i);
        }

        //-- 从末端节点回溯到根节点，生成路径
        std::vector<std::vector<edgePoint>> routeList;
        routeList.reserve(endNodes.size());
        for (size_t i : endNodes)
        {
            std::vector<edgePoint> route;
            route.reserve(32);
            route.push_back(mvPoints[i]);

            while (true)
            {
                int father_id = route.back().father_id;
                //-- 找到根节点就退
                if (father_id < 0)
                    break;

                // 区域生长按数组下标连续编号，所以 father_id 就是数组下标。
                if (father_id >= static_cast<int>(mvPoints.size()))
                    break;
                route.push_back(mvPoints[father_id]);
            } //-- 得到了从末端节点出发，到根节点停止的一条边缘
            routeList.push_back(std::move(route));
        } //-- 得到了从各个末端节点出发，到根节点停止的边缘

        if (routeList.empty())
            return std::vector<edgePoint>();

        //-- 将routeList按每个路径长度的大小排列
        std::sort(routeList.begin(), routeList.end(),
                  [](const auto &a, const auto &b)
                  { return a.size() > b.size(); });

        //-- 如果只有一个列表，说明根节点就在边缘，此时不用merge
        if (routeList.size() == 1)
        {
            return routeList[0];
        }

        //-- 此时不止一个列表，说明有merge, 则根据根节点是不是端点区分
        //-- 根节点是端点就返回最长的，根节点不是端点就返回merge最长的

        //-- 搜根节点，得到根节点的 index
        auto root_it = std::find_if(mvPoints.begin(), mvPoints.end(),
                                    [](const edgePoint &p)
                                    { return p.isRoot; });
        if (root_it == mvPoints.end())
            return routeList[0];
        size_t root_idx = std::distance(mvPoints.begin(), root_it);

        if (mvPoints[root_idx].degree == 1)
        {
            //-- 如果根节点度是1，那么它就是末端节点，那么也只是返回一条，此时返回最长的route
            return routeList[0];
        }
        else if (mvPoints[root_idx].degree > 1)
        {
            //-- 如果根节点度数大于1，那么它需要merge，此时merge两个不重合部分最长的route
            size_t idx_0 = 0, idx_1 = 0;
            int maxLength = -1;
            for (size_t j = 0; j < routeList.size(); ++j)
            {
                for (size_t k = j + 1; k < routeList.size(); ++k)
                {
                    //-- 若 overlap 初始化为0, 只有根节点相同则 overlap = 2，两个根节点father 都是-1，两个次节点都指向根节点
                    //-- 第 i+1 组点的 father_id 不同说明第 i 个点不同, overlap 只有 i-1
                    //-- 所以 overlap 要初始化为 -1，因为father_id不同指的是上一组点就不同
                    int overlap = -1;
                    // 从末端向根节点遍历（逆向比较）
                    auto it_a = routeList[j].rbegin();
                    auto it_b = routeList[k].rbegin();
                    while (it_a != routeList[j].rend() && it_b != routeList[k].rend() && it_a->father_id == it_b->father_id)
                    {
                        ++overlap;
                        ++it_a;
                        ++it_b;
                    }

                    int size = routeList[j].size() + routeList[k].size() - overlap;
                    //-- 两个route不相交且长度最长（相当于丢弃了其他的branch）
                    if (overlap == 1 && size > maxLength)
                    {
                        idx_0 = j;
                        idx_1 = k;
                        maxLength = size;
                    }
                }
            }

            if (maxLength < 0)
                return routeList[0];

            //-- 现在得到idx_0, idx_1两个routeList加在一起长度最长，且不相交，将这两个部分合并起来
            //-- 一个正序，一个逆序，避开末端点
            routeList[idx_1].insert(
                routeList[idx_1].end(),
                routeList[idx_0].rbegin() + 1, routeList[idx_0].rend());

            return routeList[idx_1];
        }

        return routeList[0];
    }

    // //-- 计算每个节点的入度
    void EdgeCluster::calculateDegree()
    {
        //-- 批量清零入度
        for (auto &point : mvPoints)
        {
            point.degree = 0;
        }

        //-- 遍历查找每个点的父节点并更新入度
        for (const auto &point : mvPoints)
        {
            //-- 根节点的 father_id 是 0
            if (point.father_id < 0)
                continue;

            if (point.father_id < static_cast<int>(mvPoints.size()))
                mvPoints[point.father_id].degree++;
        }
    }
}
