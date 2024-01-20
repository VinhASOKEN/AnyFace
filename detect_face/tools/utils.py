import numpy as np

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"[{self.x}, {self.y}]"

class Rect(object):
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def intersection_area(self, b):
        x1 = np.maximum(self.x, b.x)
        y1 = np.maximum(self.y, b.y)
        x2 = np.minimum(self.x + self.w, b.x + b.h)
        y2 = np.minimum(self.y + self.w, b.y + b.h)
        return np.abs(x1 - x2) * np.abs(y1 - y2)

class FaceObject(object):
    def __init__(self, source_dict = None):
        if source_dict is not None:
            self.prob = source_dict["prob"]
            rect = source_dict["rect"]
            self.rect = Rect(x=rect["x"], 
                             y=rect["y"], 
                             w=rect["w"], 
                             h=rect["h"])
            landmark = source_dict["landmark"]
            self.landmark = [Point(x = point_i["x"], y = point_i["y"]) for point_i in landmark]
        else:    
            self.prob = 0.0
            self.rect = Rect()
            self.landmark = []

    def todict(self):
        return {
            "prob": self.prob,
            "rect": {
                "x": self.rect.x,
                "y": self.rect.y,
                "w": self.rect.w,
                "h": self.rect.h,
            },
            "landmark": [
                {"x": point_i.x,
                 "y": point_i.y} for point_i in self.landmark
            ]
        }