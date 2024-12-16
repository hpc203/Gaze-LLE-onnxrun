#ifndef GAZE_H
#define GAZE_H
#include "facedet.h"


class GazeLLE
{
public:
	GazeLLE(std::string model_path);
	std::vector<cv::Mat> predict(const cv::Mat& srcimg, const std::vector<Bbox>& head_boxes);   
private:
	void preprocess(const cv::Mat& img);
    std::vector<float> input_image;
	std::vector<float> head_boxes_xyxy_norm;

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Gaze Target Estimation");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	Ort::Session *ort_session = nullptr;
	const std::vector<const char*> input_names = {"image_bgr", "bboxes_x1y1x2y2"};
	std::vector<const char*> output_names;
	const int input_size = 448;     
	std::vector<std::vector<int64_t>> output_node_dims;

	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::RunOptions runOptions;
};

void draw_gaze(cv::Mat& frame, const std::vector<Bbox>& head_boxes, const std::vector<cv::Mat>& heatmaps, const float thr=0.3f);

#endif