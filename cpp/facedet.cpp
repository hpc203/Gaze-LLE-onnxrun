#include "facedet.h"


using namespace cv;
using namespace std;
using namespace Ort;


static float GetIoU(const Bbox box1, const Bbox box2)
{
    float x1 = max(box1.xmin, box2.xmin);
    float y1 = max(box1.ymin, box2.ymin);
    float x2 = min(box1.xmax, box2.xmax);
    float y2 = min(box1.ymax, box2.ymax);
    float w = max(0.f, x2 - x1);
    float h = max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

static void nms_boxes(vector<Bbox> &boxes, const float nms_thresh)
{
    sort(boxes.begin(), boxes.end(), [](Bbox a, Bbox b)
         { return a.score > b.score; });
    const int num_box = boxes.size();
    vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr >= nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    int idx_t = 0;
    boxes.erase(remove_if(boxes.begin(), boxes.end(), [&idx_t, &isSuppressed](const Bbox &f)
                          { return isSuppressed[idx_t++]; }),
                boxes.end());
}

FaceDet::FaceDet(string model_path)
{
    if (model_path.empty()) 
    {
        std::cout << "onnx path error" << std::endl;
    }

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    // 加载模型
    std::wstring_convert <std::codecvt_utf8<wchar_t>> converter;
#ifdef _WIN32
    std::wstring w_model_path = converter.from_bytes(model_path);
    ort_session = new Ort::Session(env, w_model_path.c_str(), sessionOptions);
#else
    ort_session = new Ort::Session(env, model_path.c_str(), sessionOptions);
#endif

    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    
    for (int i = 0; i < numOutputNodes; i++)
    {
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    } 
}

void FaceDet::preprocess(const cv::Mat& srcimg)
{
    const float im_ratio = float(srcimg.rows) / (float)srcimg.cols;
    int new_width = this->input_size;
    int new_height = int(new_width * im_ratio);
    if(im_ratio>1)
    {
        new_height = this->input_size;
        new_width = int(new_height / im_ratio);
    }
    this->det_scale = float(new_height) / (float)srcimg.rows;
    Mat resized_img;
    resize(srcimg, resized_img, Size(new_width, new_height));
    Mat det_img;
    copyMakeBorder(resized_img, det_img, 0, this->input_size - new_height, 0, this->input_size - new_width, BORDER_CONSTANT, 0);

    vector<cv::Mat> bgrChannels(3);
    split(det_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int image_area = this->input_size * this->input_size;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

void FaceDet::generate_proposal(const float* p_box, const float* p_scores, const float* p_kps, const int stride, vector<Bbox>& boxes)
{
	const int feat_h = this->input_size / stride;
	const int feat_w = this->input_size / stride;
    const int num_anchors = 2;
	for (int i = 0; i < feat_h; i++)
	{
		for (int j = 0; j < feat_w; j++)
		{
			for(int n=0; n<num_anchors; n++)
            {
                const int index = i * feat_w*num_anchors + j*num_anchors+n;
                if(p_scores[index] >= this->det_thresh)
                {
                    Bbox box;
                    box.xmin = (j - p_box[index * 4]) * stride;
                    box.ymin = (i - p_box[index * 4 + 1]) * stride;
                    box.xmax = (j + p_box[index * 4 + 2]) * stride;
                    box.ymax = (i + p_box[index * 4 + 3]) * stride;
                    box.xmin /= this->det_scale;
                    box.ymin /= this->det_scale;
                    box.xmax /= this->det_scale;
                    box.ymax /= this->det_scale;

                    for(int k=0;k<5;k++)
                    {
                        float px = (j + p_kps[index * 10 + k * 2]) * stride;
                        float py = (i + p_kps[index * 10 + k * 2 + 1]) * stride;
                        px /= this->det_scale;
                        py /= this->det_scale;
                        box.kps[k * 2] = px;
                        box.kps[k * 2 + 1] = py;
                    }
                    box.score = p_scores[index];
                    boxes.emplace_back(box);
                }

            }
		}
	}
}


vector<Bbox> FaceDet::detect(const Mat& srcimg)
{
    this->preprocess(srcimg);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_size, this->input_size};  ///也可以写在构造函数里
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), this->output_names.size());
    vector<Bbox> boxes;
    for(int i=0;i<3;i++)
    {
        float *p_scores = ort_outputs[i].GetTensorMutableData<float>();
        float *p_bbox = ort_outputs[i + this->fmc].GetTensorMutableData<float>();
        float *p_kps = ort_outputs[i + this->fmc*2].GetTensorMutableData<float>();
        
        this->generate_proposal(p_bbox, p_scores, p_kps, this->feat_stride_fpn[i], boxes);
    }
    nms_boxes(boxes, this->nms_thresh);

    for(int i=0;i<boxes.size();i++)
    {
        const float padw = (boxes[i].xmax - boxes[i].xmin) * 0.15;
        const float padh = (boxes[i].ymax - boxes[i].ymin) * 0.15;
        boxes[i].xmin -= padw;
        boxes[i].ymin -= padh;
        boxes[i].xmax += padw;
        boxes[i].ymax += padh;
    }
    return boxes;
}