from itertools import product as product
import numpy as np
import cv2
from skimage import transform as trans

class Aligner():
    def __init__(self):
        pass

    def align_face(self, img, bbox=None, landmark=None, image_size=(112, 112)):
        M = None
        if landmark is not None:
            assert len(image_size) == 2
            src = np.array([
                [30.2946 * image_size[0] / 112, 51.6963 * image_size[1] / 112],
                [65.5318 * image_size[0] / 112, 51.5014 * image_size[1] / 112],
                [48.0252 * image_size[0] / 112, 71.7366 * image_size[1] / 112],
                [33.5493 * image_size[0] / 112, 92.3655 * image_size[1] / 112],
                [62.7299 * image_size[0] / 112, 92.2041 * image_size[1] / 112]], dtype=np.float32)

            src[:, 0] += 8.0 * image_size[1] / 112
            dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]
        
        assert M is not None
        if M is None:
            if bbox is None:
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1] * 0.0625)
                det[1] = int(img.shape[0] * 0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = 44 * image_size[0] / 112
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
            ret = img[bb[1]:bb[3], bb[0]:bb[2], :]

            if len(image_size) > 0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret
        else:
            assert len(image_size) == 2
            warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
            return warped

    def __call__(self, img, faceobject, image_size = (112, 112)):
        x = faceobject.rect.x
        y = faceobject.rect.y
        w = faceobject.rect.w
        h = faceobject.rect.h
        lm_x1 = faceobject.landmark[0].x
        lm_y1 = faceobject.landmark[0].y
        lm_x2 = faceobject.landmark[1].x
        lm_y2 = faceobject.landmark[1].y
        lm_x3 = faceobject.landmark[2].x
        lm_y3 = faceobject.landmark[2].y
        lm_x4 = faceobject.landmark[3].x
        lm_y4 = faceobject.landmark[3].y
        lm_x5 = faceobject.landmark[4].x
        lm_y5 = faceobject.landmark[4].y
        bbox = np.array([x, y, w, h])
        landmark = np.array([[lm_x1, lm_y1],
                             [lm_x2, lm_y2],
                             [lm_x3, lm_y3],
                             [lm_x4, lm_y4],
                             [lm_x5, lm_y5]])
        face = self.align_face(img, bbox, landmark, image_size)
        return face

    def AlignFace(self, img, faceobject, image_size = (112, 112)):
        return self.__call__(img, faceobject, image_size)