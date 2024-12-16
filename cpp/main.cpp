#include "gazelle.h"
#include<opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main()
{
    FaceDet face_det("/home/wangbo/gaze-lle/weights/det_face.onnx");   /////注意文件路径要写对
    GazeLLE gazelle("/home/wangbo/gaze-lle/weights/gazelle_dinov2_vitb14_inout_1x3x448x448_1xNx4.onnx");
    string imgpath = "/home/wangbo/gaze-lle/testimgs/jim-and-dwights-customer-service-training-1627594561.jpg";

    Mat srcimg = imread(imgpath);
    vector<Bbox> head_boxes = face_det.detect(srcimg);
    vector<Mat> resized_heatmaps = gazelle.predict(srcimg, head_boxes);

    draw_gaze(srcimg, head_boxes, resized_heatmaps);

    namedWindow("Gaze Target Estimation", WINDOW_NORMAL);
    imshow("Gaze Target Estimation", srcimg);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
