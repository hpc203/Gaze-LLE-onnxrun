#ifndef FACE_DETECT_H
#define FACE_DETECT_H
#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    float kps[10];
} Bbox;

class FaceDet
{
public:
	FaceDet(std::string model_path);
	std::vector<Bbox> detect(const cv::Mat& srcimg);   
private:
	void preprocess(const cv::Mat& img);
	std::vector<float> input_image;
	float det_scale;
	void generate_proposal(const float* p_box, const float* p_scores, const float* p_kps, const int stride, std::vector<Bbox>& boxes);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Detect");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	Ort::Session *ort_session = nullptr;
	const std::vector<const char*> input_names = {"input.1"};
	const std::vector<const char*> output_names = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};
	const int input_size = 512;
	std::vector<std::vector<int64_t>> output_node_dims;
	const float det_thresh = 0.5;
    const int fmc = 3;
    const int feat_stride_fpn[3] = {8, 16, 32};
	const float nms_thresh = 0.4;

	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::RunOptions runOptions;
};

#endif