import onnxruntime 
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class GazeLLE:
    def __init__(self, model_path):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(model_path, so)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.input_size = 448

    def preprocess(self, image):
        input_img = cv2.resize(image, (self.input_size, self.input_size))
        input_img = np.transpose(input_img, (2, 0, 1))
        return np.expand_dims(input_img.astype(np.float32), axis=0)
    
    def predict(self, srcimg, head_boxes, disable_attention_heatmap_mode=True):
        img_h, img_w = srcimg.shape[:2]
        input_img = self.preprocess(srcimg)

        head_boxes_xyxy_norm = head_boxes[:, :4] / np.array([[img_w, img_h, img_w, img_h]], dtype=np.float32)
        head_boxes_xyxy_norm = np.expand_dims(head_boxes_xyxy_norm.astype(np.float32), axis=0)
        outputs = self.session.run(None, {self.input_names[0]: input_img, self.input_names[1]: head_boxes_xyxy_norm})
        
        heatmaps = outputs[0]
        if len(outputs) == 2:
            inout = outputs[1]    ####置信度score,这个没那么重要，在画图的时候没有画上去,可以自己加上去
        # PostProcess
        result_image, resized_heatmaps = self.postprocess(srcimg.copy(), heatmaps, disable_attention_heatmap_mode)
        return result_image, resized_heatmaps

    def postprocess(self, image_bgr, heatmaps, disable_attention_heatmap_mode):
        image_height = image_bgr.shape[0]
        image_width = image_bgr.shape[1]
        if not disable_attention_heatmap_mode:
            image_rgb = image_bgr[..., ::-1]
            heatmaps_all: np.ndarray = np.sum(heatmaps, axis=0) # [64, 64]
            heatmaps_all = heatmaps_all * 255
            heatmaps_all = heatmaps_all.astype(np.uint8)
            heatmaps_all = cv2.resize(heatmaps_all, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            heatmaps_all = plt.cm.jet(heatmaps_all / 255.0)
            heatmaps_all = (heatmaps_all[:, :, :3] * 255).astype(np.uint8)
            heatmaps_all = Image.fromarray(heatmaps_all).convert("RGBA")
            heatmaps_all.putalpha(128)
            image_rgba = Image.alpha_composite(Image.fromarray(image_rgb).convert("RGBA"), heatmaps_all)
            image_bgr = cv2.cvtColor(np.asarray(image_rgba)[..., [2,1,0,3]], cv2.COLOR_BGRA2BGR)

        heatmap_list = [cv2.resize(heatmap[..., None], (image_width, image_height)) for heatmap in heatmaps]
        resized_heatmaps = np.asarray(heatmap_list)

        return image_bgr, resized_heatmaps
    

def calculate_centroid(heatmap):
    max_index = np.argmax(heatmap)
    y, x = np.unravel_index(max_index, heatmap.shape)
    return int(x), int(y), heatmap[y, x]

def draw_gaze(frame, head_boxes, heatmaps, thr=0.30):
    for head_box, heatmap in zip(head_boxes, heatmaps):
        cx, cy, score = calculate_centroid(heatmap)
        if score >= thr:
            head_cx, head_cy = int((head_box[0]+head_box[2])*0.5), int((head_box[1]+head_box[3])*0.5)
            cv2.line(frame, (head_cx, head_cy), (cx, cy), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.line(frame, (head_cx, head_cy), (cx, cy), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    return frame