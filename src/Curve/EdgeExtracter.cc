#include "Curve/EdgeExtracter.h"
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <utility>

namespace ORB_SLAM2
{

    std::vector<Edge> EdgeExtracter::operator()(const cv::Mat &imGray, const cv::Mat &imSem)
    {
        // Edge extraction and processing
        cv::Canny(imGray, mCanny, threshold1, threshold2, 3, true);

        cv::Mat grad_x, grad_y;
        cv::Scharr(imGray, grad_x, CV_32F, 1, 0);
        cv::Scharr(imGray, grad_y, CV_32F, 0, 1);

        mMatGradAngle.create(imGray.size(), CV_32F);
        // 区域生长只使用梯度方向，不再计算和保存从未读取的梯度幅值。
        cv::phase(grad_x, grad_y, mMatGradAngle, true);

        preprocessSem(imSem);

        preprocessEdge();

        regionGrowthClusteringOCanny(angleThreshold);

        cvt2OrderedEdges();

        return std::move(mvEdges);
    }

    inline float EdgeExtracter::calcAngleBias(float angle_1, float angle_2)
    {
        float res = fabs(angle_1 - angle_2);
        if (res > 180)
        {
            res = 360 - res;
        }
        return res;
    }

    void EdgeExtracter::preprocessSem(const cv::Mat &imSem)
    {
        if (imSem.empty())
        {
            // 空矩阵表示所有像素属于同一个有效类别，省去每帧整图填充。
            mSem.release();
            return;
        }

        cv::Mat filteredSem = cv::Mat::zeros(imSem.size(), CV_8UC1);
        const cv::Mat morphologyKernel = cv::Mat::ones(5, 5, CV_8UC1);
        for (int i = 1; i < 10; ++i)
        {
            cv::Mat temp;
            cv::compare(imSem, i, temp, cv::CMP_EQ);

            if (cv::countNonZero(temp) == 0)
                continue;

            cv::Mat label, stats, centroids;
            cv::connectedComponentsWithStats(temp, label, stats, centroids, 4, CV_16U);

            for (int j = 1; j < centroids.rows; j++)
            {
                int area = stats.at<int>(j, cv::CC_STAT_AREA);
                int width = stats.at<int>(j, cv::CC_STAT_WIDTH);
                int height = stats.at<int>(j, cv::CC_STAT_HEIGHT);

                if (area < 200)
                    continue;

                double WHratio = std::max(width, height) / std::min(width, height);
                if (WHratio > 5)
                    continue;

                cv::Mat mask;
                cv::compare(label, j, mask, cv::CMP_EQ);
                cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, morphologyKernel);
                cv::morphologyEx(mask, mask, cv::MORPH_OPEN, morphologyKernel);
                cv::dilate(mask, mask, morphologyKernel);

                filteredSem.setTo(i, mask);
            }
        }
        mSem = filteredSem;
    }

    void EdgeExtracter::preprocessEdge()
    {
        for (int i = 0; i < mCanny.rows; ++i)
        {
            uint8_t *current = mCanny.ptr<uint8_t>(i);
            uint8_t *above = i > 0 ? mCanny.ptr<uint8_t>(i - 1) : nullptr;
            uint8_t *below = i < mCanny.rows - 1 ? mCanny.ptr<uint8_t>(i + 1) : nullptr;

            for (int j = 0; j < mCanny.cols; ++j)
            {
                if (current[j] == 0)
                    continue; // 跳过非边缘点

                int left = j > 0 ? current[j - 1] : 0;
                int right = j < mCanny.cols - 1 ? current[j + 1] : 0;
                int up = above ? above[j] : 0;
                int down = below ? below[j] : 0;

                bool connected = (left > 0 && up > 0) || (right > 0 && up > 0) ||
                                 (left > 0 && down > 0) || (right > 0 && down > 0);

                if (connected)
                {
                    current[j] = 0;
                }
            }
        }
    }

    void EdgeExtracter::cvt2OrderedEdges()
    {
        // 并行写入，因此直接resize而不是reverse
        mvEdges.resize(mvEdgeClusters.size());

        // 使用TBB并行处理每个边缘簇
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mvEdgeClusters.size()),
                          [&](const tbb::blocked_range<size_t> &range)
                          {
                              for (size_t i = range.begin(); i != range.end(); ++i)
                              {
                                  Edge curr_edge;
                                  const auto &cluster_points = mvEdgeClusters[i].organize();
                                  curr_edge.reserve(cluster_points.size());

                                  for (const auto &point : cluster_points)
                                  {
                                      const int x = static_cast<int>(point.pixel.x);
                                      const int y = static_cast<int>(point.pixel.y);

                                      curr_edge.emplace_back(x, y);
                                  }
                                  mvEdges[i] = std::move(curr_edge); // 直接写入到预分配的位置
                              }
                          });
    }

    void EdgeExtracter::regionGrowthClusteringOCanny(float angle_Thres, const cv::Point &offset)
    {
        cv::Mat visitedMat(mCanny.rows, mCanny.cols, CV_8UC1, cv::Scalar::all(0));
        mvEdgeClusters.clear();

        uint8_t *canny_ptr = mCanny.data;
        const bool useSemanticMask = !mSem.empty();
        uint8_t *sem_ptr = useSemanticMask ? mSem.data : nullptr;
        uint8_t *visited_ptr = visitedMat.data;
        const float *angle_ptr = mMatGradAngle.ptr<float>();
        const int canny_step = static_cast<int>(mCanny.step);
        const int sem_step = static_cast<int>(mSem.step);
        const int visited_step = static_cast<int>(visitedMat.step);
        const int angle_step = static_cast<int>(mMatGradAngle.step / sizeof(float));
        const int width = mCanny.cols;
        const int height = mCanny.rows;

        static const int kDx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        static const int kDy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (visited_ptr[y * visited_step + x] != 0 ||
                    canny_ptr[y * canny_step + x] != 255 ||
                    (useSemanticMask && sem_ptr[y * sem_step + x] == 0))
                    continue;

                std::vector<edgePoint> current_cluster;
                current_cluster.reserve(128);
                visited_ptr[y * visited_step + x] = 1;
                int cls = useSemanticMask ? sem_ptr[y * sem_step + x] : 1;

                edgePoint curr_edge_point(cv::Point(x, y), -1);
                curr_edge_point.isRoot = true;

                std::vector<edgePoint> open_list;
                open_list.reserve(256);
                open_list.push_back(curr_edge_point);
                size_t head = 0;

                while (head < open_list.size())
                {
                    const int currentPointId = static_cast<int>(head);
                    const edgePoint current_point = open_list[head++];

                    const int cx = current_point.pixel.x;
                    const int cy = current_point.pixel.y;

                    current_cluster.push_back(current_point);

                    const float curr_angle = angle_ptr[(cy + offset.y) * angle_step + (cx + offset.x)];

                    for (int k = 0; k < 8; ++k)
                    {
                        const int nx = cx + kDx[k];
                        const int ny = cy + kDy[k];
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height ||
                            (useSemanticMask && sem_ptr[ny * sem_step + nx] != cls))
                        {
                            continue;
                        }

                        const int visited_idx = ny * visited_step + nx;
                        if (visited_ptr[visited_idx] != 0)
                        {
                            continue;
                        }

                        if (canny_ptr[ny * canny_step + nx] != 255)
                        {
                            continue;
                        }

                        const float neigh_angle = angle_ptr[(ny + offset.y) * angle_step + (nx + offset.x)];

                        float angle_bias = calcAngleBias(neigh_angle, curr_angle);
                        if (angle_bias >= angle_Thres)
                        {
                            continue;
                        }

                        visited_ptr[visited_idx] = 1;
                        open_list.emplace_back(cv::Point(nx, ny), currentPointId);
                    }
                }

                if (current_cluster.size() > 10)
                {
                    mvEdgeClusters.emplace_back(std::move(current_cluster));
                }
            }
        }
    }
}
