import cv2
from det_face import FaceDet
from gazelle import GazeLLE, draw_gaze


if __name__=='__main__':
    # Load the face detection model
    face_det = FaceDet('weights/det_face.onnx')
    # Load the gaze estimation model
    gazelle = GazeLLE('weights/gazelle_dinov2_vitb14_inout_1x3x448x448_1xNx4.onnx')
    # Load the image
    imgpath = "testimgs/jim-and-dwights-customer-service-training-1627594561.jpg"
    srcimg = cv2.imread(imgpath)
    # Detect faces in the image
    head_boxes, _ = face_det.detect(srcimg)
    # Predict the gaze of the detected faces
    result_image, resized_heatmaps = gazelle.predict(srcimg, head_boxes, disable_attention_heatmap_mode=True)
    result_image = draw_gaze(result_image, head_boxes, resized_heatmaps)

    cv2.namedWindow('Gaze Target Estimation', cv2.WINDOW_NORMAL)
    cv2.imshow('Gaze Target Estimation', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()