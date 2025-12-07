#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <algorithm>

// 函数声明
void processContours(cv::Mat const &imgMask, cv::Mat &imgDisplay, std::string const &label);
void preprocessImage(cv::Mat const &imgInput, cv::Mat &imgMask, cv::Scalar const &lower, cv::Scalar const &upper);

int main()
{
    try
    {
        std::string videoPath = "/home/zoupeng/Dart_workspace/src/dart_vision_pkg/samples/test2.webm";
        cv::VideoCapture cap(videoPath);

        cv::Mat img, imgGreenMask;

        // 绿色范围 (HSV)
        cv::Scalar lowerGreen(40, 80, 80);
        cv::Scalar upperGreen(80, 255, 255);

        while (true)
        {
            cap.read(img);
            if (img.empty())
                break;

            // 1. 预处理
            preprocessImage(img, imgGreenMask, lowerGreen, upperGreen);

            // 2. 轮廓查找 + 筛选最大且比例合适的轮廓
            processContours(imgGreenMask, img, "Target");

            // 显示结果
            cv::imshow("Result Image", img);
            cv::imshow("Green Mask", imgGreenMask);

            if (cv::waitKey(10) == 27)
                break; // ESC 退出
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what();
    }

    return 0;
}

/**
 * @brief 预处理函数
 */
void preprocessImage(cv::Mat const &imgInput, cv::Mat &imgMask, cv::Scalar const &lower, cv::Scalar const &upper)
{
    cv::Mat imgHSV;
    cv::GaussianBlur(imgInput, imgInput, cv::Size(5, 5), 0);
    cv::cvtColor(imgInput, imgHSV, cv::COLOR_BGR2HSV);
    cv::inRange(imgHSV, lower, upper, imgMask);

    // 膨胀 + 闭合
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(imgMask, imgMask, kernel);
    cv::morphologyEx(imgMask, imgMask, cv::MORPH_CLOSE, kernel);
}

/**
 * @brief 轮廓处理函数：筛选长宽比 <= 1.25 且 面积最大 的轮廓
 */
void processContours(cv::Mat const &imgMask, cv::Mat &imgDisplay, std::string const &label)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // 查找轮廓
    cv::findContours(imgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // --- 变量初始化：用于记录最佳轮廓 ---
    int bestContourIdx = -1;
    double maxArea = 0.0;
    cv::Rect bestBoundRect;

    // 1. 遍历所有轮廓，寻找最佳候选
    for (size_t i = 0; i < contours.size(); i++)
    {
        // 获取外接矩形
        cv::Rect boundRect = cv::boundingRect(contours[i]);

        // 基础尺寸检查（防止只有1个像素的噪点干扰）
        if (boundRect.width < 3 || boundRect.height < 3)
            continue;

        // 计算长宽比
        float width = static_cast<float>(boundRect.width);
        float height = static_cast<float>(boundRect.height);
        float aspectRatio = std::max(width, height) / std::min(width, height);

        // 筛选条件 1: 长宽比符合要求 (接近圆形/正方形)
        if (aspectRatio <= 1.25f)
        {
            // 计算面积
            double area = cv::contourArea(contours[i]);

            // 筛选条件 2: 面积必须是目前见过的最大的
            if (area > maxArea)
            {
                maxArea = area;
                bestContourIdx = (int)i;
                bestBoundRect = boundRect;
            }
        }
    }

    // 2. 如果找到了最佳轮廓，再进行绘制
    if (bestContourIdx != -1)
    {
        // 轮廓平滑 (仅对最终选定的轮廓操作，节省性能)
        std::vector<cv::Point> conPoly;
        float peri = cv::arcLength(contours[bestContourIdx], true);
        cv::approxPolyDP(contours[bestContourIdx], conPoly, 0.02 * peri, true);

        // 绘制外接矩形
        cv::rectangle(imgDisplay, bestBoundRect.tl(), bestBoundRect.br(), cv::Scalar(0, 255, 0), 2);

        // 绘制文字
        cv::putText(imgDisplay, label, {bestBoundRect.x, bestBoundRect.y - 5},
                    cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 255, 0), 1);

        // 打印最大面积数值
        // std::cout << "Max Area: " << maxArea << std::endl;
    }
}