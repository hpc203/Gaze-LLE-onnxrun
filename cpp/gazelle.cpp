#include "gazelle.h"


using namespace cv;
using namespace std;
using namespace Ort;


GazeLLE::GazeLLE(string model_path)
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
        // AllocatedStringPtr output_name_Ptr = ort_session->GetOutputNameAllocated(i, allocator);
        // string output_node_name = output_name_Ptr.get();
        // output_names.push_back(output_node_name.c_str());
        // cout<<"output_names["<<i<<"]:"<<this->output_names[i]<<endl;
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    if(numOutputNodes==2)
    {
        this->output_names = {"heatmap", "inout"};
    }
    else
    {
        this->output_names = {"heatmap"};
    }
}

void GazeLLE::preprocess(const cv::Mat&  img)
{
    Mat resized_img;
    resize(img, resized_img, Size(this->input_size, this->input_size));

    vector<cv::Mat> bgrChannels(3);
    split(resized_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1);
    }

    const int image_area = this->input_size * this->input_size;
    this->input_image.resize(1*3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

vector<Mat> GazeLLE::predict(const cv::Mat& srcimg, const std::vector<Bbox>& head_boxes)
{
    const float img_h = (float)srcimg.rows;
    const float img_w = (float)srcimg.cols;
    this->preprocess(srcimg);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_size, this->input_size};
    const int num_box = head_boxes.size();
    std::vector<int64_t> input_head_boxes_shape = {1, (int64_t)num_box, 4};   ////不考虑batchsize,一直等于1
    this->head_boxes_xyxy_norm.clear();
    this->head_boxes_xyxy_norm.resize(1*num_box*4);
    for(int i=0;i<num_box;i++)
    {
        this->head_boxes_xyxy_norm[i*4] = head_boxes[i].xmin / img_w;
        this->head_boxes_xyxy_norm[i*4+1] = head_boxes[i].ymin / img_h;
        this->head_boxes_xyxy_norm[i*4+2] = head_boxes[i].xmax / img_w;
        this->head_boxes_xyxy_norm[i*4+3] = head_boxes[i].ymax / img_h;
    }

    vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, this->head_boxes_xyxy_norm.data(), this->head_boxes_xyxy_norm.size(), input_head_boxes_shape.data(), input_head_boxes_shape.size()));

    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), inputTensors.data(), inputTensors.size(), this->output_names.data(), this->output_names.size());

    float *pdata = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_map = out_shape[0];
    vector<Mat> resized_heatmaps(num_map);
    const int image_area = out_shape[1]*out_shape[2];
    for(int i=0;i<num_map;i++)
    {
        Mat heatmap = Mat(out_shape[1], out_shape[2], CV_32FC1, pdata+i*image_area);
        resize(heatmap, resized_heatmaps[i], Size(srcimg.cols, srcimg.rows));
    }
    
    // if(ort_outputs.size()==2)
    // {
    //     float* inout = ort_outputs[1].GetTensorMutableData<float>();
    // }
    ////postprocess
    /////不做disable_attention_heatmap_mode，画出眼睛注视的线段更重要
    return resized_heatmaps;
}

void draw_gaze(cv::Mat& frame, const std::vector<Bbox>& head_boxes, const std::vector<cv::Mat>& heatmaps, const float thr)
{
    const int num_box = head_boxes.size();
    for(int i=0;i<num_box;i++)
    {
        double max_score;;
        Point classIdPoint;
        minMaxLoc(heatmaps[i], 0, &max_score, 0, &classIdPoint);
        const int cx = classIdPoint.x;
        const int cy = classIdPoint.y;
        if(max_score >= thr)
        {
            const int head_cx = int((head_boxes[i].xmin+head_boxes[i].xmax)*0.5);
            const int head_cy = int((head_boxes[i].ymin+head_boxes[i].ymax)*0.5);
            cv::line(frame, Point(head_cx, head_cy), Point(cx, cy), Scalar(255, 255, 255), 3, LINE_AA);
            cv::line(frame, Point(head_cx, head_cy), Point(cx, cy), Scalar(0, 255, 0), 2, LINE_AA);
            cv::circle(frame, Point(cx, cy), 4, Scalar(255, 255, 255), -1, LINE_AA);
            cv::circle(frame, Point(cx, cy), 3, Scalar(0, 0, 255), -1, LINE_AA);
        }
    }
}