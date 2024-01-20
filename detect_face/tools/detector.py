from itertools import product as product
from typing import Union, Tuple, List
from math import ceil, sqrt
import numpy as np
import cv2
import onnxruntime
onnxruntime.set_default_logger_severity(3)
from .utils import FaceObject, Point

class PriorBox(object):
    def __init__(self, image_size: Union[int, Tuple[int, int]] = (640, 640)):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = True
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output.clip(max=1, min=0)
        return output

class FaceDetector():
    def __init__(self,
                 model_file: str = "",
                 providers: List[str] = ['CPUExecutionProvider'],
                 num_threads: int = 1,
                 prob_threshold: float = 0.7,
                 nms_threshold: float = 0.5,
                 target_size: int = 840,
                 max_size: int = 960):
        self.ort_session = onnxruntime.InferenceSession(model_file, providers=providers)
        self.ort_session.intra_op_num_threads = num_threads
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.target_size = target_size
        self.max_size = max_size
        assert self.target_size <= self.max_size, "\"max_size\" must be greater than \"target_size\""
    
    def __call__(self, image_cv2: np.ndarray):
        img_tensor = self.preprocess(image_cv2)
        ort_inputs = {
                self.ort_session.get_inputs()[0].name: np.float32(img_tensor)
            }
        
        loc, conf, landms = self.ort_session.run(None, ort_inputs)
        dets = self.postprocess(loc, conf, landms)
        faceobjects = self.dets2faceobjects(dets)
        faceobjects = self.sort_faceobjects(faceobjects, image_cv2)
        return faceobjects
    
    def DetectFace(self, image_cv2: np.ndarray):
        return self.__call__(image_cv2)
  
    def preprocess(self, image_cv2: np.ndarray):
        img = np.float32(image_cv2)

        # testing scale
        self.im_shape = img.shape
        im_size_min = np.min(self.im_shape[0:2])
        im_size_max = np.max(self.im_shape[0:2])
        self.resize = float(self.target_size) / float(im_size_min)

        if np.round(self.resize * im_size_max) > self.max_size:
            self.resize = float(self.max_size) / float(im_size_max)
        if self.resize != 1:
            img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
        self.im_height, self.im_width, _ = img.shape
        self.scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        self.scale_lm = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, loc: np.ndarray, conf: np.ndarray, landms: np.ndarray):
        priorbox = PriorBox(image_size=(self.im_height, self.im_width))
        priors = priorbox.forward()
        prior_data = priors
        boxes = self.decode(loc.squeeze(0), prior_data, (0.1, 0.2))
        boxes = boxes * self.scale / self.resize
        scores = conf.squeeze(0)[:, 1]
        landms = self.decode_landm(landms.squeeze(0), prior_data, (0.1, 0.2))
        
        landms = landms * self.scale_lm / self.resize
        
        inds = np.where(scores > self.prob_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def dets2faceobjects(self, dets: np.ndarray):
        faceobjects = []
        for det in dets:
            faceobject = FaceObject()
            faceobject.prob = det[4]
            x0 = np.maximum(np.minimum(det[0], float(self.im_shape[1]) - 1), 0.0)
            y0 = np.maximum(np.minimum(det[1], float(self.im_shape[0]) - 1), 0.0)
            x1 = np.maximum(np.minimum(det[2], float(self.im_shape[1]) - 1), 0.0)
            y1 = np.maximum(np.minimum(det[3], float(self.im_shape[0]) - 1), 0.0)
            faceobject.rect.x = x0
            faceobject.rect.y = y0
            faceobject.rect.w = x1 - x0
            faceobject.rect.h = y1 - y0
            faceobject.landmark.append(self.set_landmark_points(det[5], det[6]))
            faceobject.landmark.append(self.set_landmark_points(det[7], det[8]))
            faceobject.landmark.append(self.set_landmark_points(det[9], det[10]))
            faceobject.landmark.append(self.set_landmark_points(det[11], det[12]))
            faceobject.landmark.append(self.set_landmark_points(det[13], det[14]))
            faceobjects.append(faceobject)
        return faceobjects

    def sort_faceobjects(self, faceobjects: List[FaceObject], image_cv2: np.ndarray):
        def sort_face(faceobject_h_w: List[Tuple[FaceObject, int, int]]):
            rect_area = faceobject_h_w[0].rect.area()
            rect_x_center = faceobject_h_w[0].rect.x + faceobject_h_w[0].rect.w / 2.0
            rect_y_center = faceobject_h_w[0].rect.y + faceobject_h_w[0].rect.h / 2.0
            img_x_center = faceobject_h_w[2] / 2.0
            img_y_center = faceobject_h_w[1] / 2.0
            dis_rect_to_center = sqrt((img_x_center - rect_x_center) ** 2 + (img_y_center - rect_y_center) ** 2)
            if dis_rect_to_center == 0:
                dis_rect_to_center = 1e-10
            return rect_area / dis_rect_to_center
        
        h, w, _ = image_cv2.shape
        faceobject_h_w = [(x, h, w) for x in faceobjects]
        faceobject_h_w = sorted(faceobject_h_w, key=sort_face, reverse=True)
        faceobjects = [faceobject[0] for faceobject in faceobject_h_w]
        return faceobjects

    def set_landmark_points(self, x: float, y:float):
        point = Point()
        point.x = x
        point.y = y
        return point


    @staticmethod
    def decode(loc: np.ndarray, priors: np.ndarray, variances: Tuple[float, float]):
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def decode_landm(pre: np.ndarray, priors: np.ndarray, variances: Tuple[float, float]):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), axis=1)
        return landms

    @staticmethod
    def py_cpu_nms(dets: np.ndarray, thresh: float):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep