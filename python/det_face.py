import onnxruntime
import numpy as np
import cv2


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def nms_boxes(boxes, scores, iou_thres):
    # Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    keep = []
    for i, box_a in enumerate(boxes):
        is_keep = True
        for j in range(i):
            if not keep[j]:
                continue
            box_b = boxes[j]
            iou = bb_intersection_over_union(box_a, box_b)
            if iou >= iou_thres:
                if scores[i] > scores[j]:
                    keep[j] = False
                else:
                    is_keep = False
                    break

        keep.append(is_keep)

    return np.array(keep).nonzero()[0]

class FaceDet:
    def __init__(self, model_path):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(model_path, so)
        self.input_size = 512
        self.input_name = self.session.get_inputs()[0].name
        self.det_thresh = 0.5
        self.nms_thresh = 0.4        
    
    def preprocess(self, srcimg):
        im_ratio = float(srcimg.shape[0]) / srcimg.shape[1]
        if im_ratio > 1:
            new_height = self.input_size
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size
            new_height = int(new_width * im_ratio)
        self.det_scale = float(new_height) / srcimg.shape[0]
        resized_img = cv2.resize(srcimg, (new_width, new_height))
        input_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        input_img[:new_height, :new_width, :] = resized_img

        input_img = (input_img - 127.5) / 128
        input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(input_img.astype(np.float32), axis=0)
    
    def detect(self, srcimg):
        input_img = self.preprocess(srcimg)

        output = self.session.run(None, {self.input_name: input_img})

        scores_list = []
        bboxes_list = []
        kpss_list = []
        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        center_cache = {}
        for idx, stride in enumerate(feat_stride_fpn):
            scores = output[idx]
            bbox_preds = output[idx + fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = output[idx + fmc * 2] * stride
            height = self.input_size // stride
            width = self.input_size // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                num_anchors = 2
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / self.det_scale
        kpss = np.vstack(kpss_list) / self.det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = nms_boxes(pre_det, [1 for s in pre_det], self.nms_thresh)
        bboxes = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if bboxes.shape[0] == 0:
            return [],[]
        for i in range(bboxes.shape[0]):
            padh, padw = (bboxes[i, 3] - bboxes[i, 1])*0.15, (bboxes[i, 2] - bboxes[i, 0])*0.15
            bboxes[i,:4] += np.array([-padw, -padh, padw, padh])  ###严格意义上来讲,应该是检测人头然后把人头检测框输入到gazelle里，人头框里包含人脸,因此把这里的人脸检测框往外扩展一部分
        return bboxes, kpss
    

if __name__=='__main__':
    facemodel = FaceDet('weights/det_face.onnx')
    imgpath = "testimgs/0.jpg"
    srcimg = cv2.imread(imgpath)
    bboxes, kpss = facemodel.detect(srcimg)

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        kps = kpss[i]
        cv2.rectangle(srcimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # for j in range(kps.shape[0]):
        #     cv2.circle(srcimg, (int(kps[j][0]), int(kps[j][1])), 3, (0, 0, 255), -1)
    
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()