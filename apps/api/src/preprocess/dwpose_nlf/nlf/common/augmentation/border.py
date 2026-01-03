import cv2
import numpy as np


def augment_border(im, border_value, rng):
    h, w = im.shape[:2]
    im = im.copy()

    p = 0.2
    size = 0.08
    angle = np.deg2rad(8)

    def random_angle():
        if rng.uniform(0, 1) < 0.9:
            return rng.uniform(-angle, angle)
        else:
            return rng.uniform(-angle, angle) * 1.5

    # top:
    if rng.uniform(0, 1) < p:
        alpha = random_angle()
        d = rng.uniform(0, h * size)
        y1 = - np.tan(alpha) * h / 2 + d
        y2 = + np.tan(alpha) * h / 2 + d
        y3 = min(0, y1)
        y4 = min(0, y2)
        points = np.array([[0, y1], [w, y2], [w, y3], [0, y4]], np.int32)
        cv2.fillPoly(im, [points], border_value, lineType=cv2.LINE_AA)

    # bottom:
    if rng.uniform(0, 1) < p:
        alpha = random_angle()
        d = rng.uniform(0, h * size)
        y1 = h - np.tan(alpha) * h / 2 - d
        y2 = h + np.tan(alpha) * h / 2 - d
        y3 = max(h, y1)
        y4 = max(h, y2)
        points = np.array([[0, y1], [w, y2], [w, y3], [0, y4]], np.int32)
        cv2.fillPoly(im, [points], border_value, lineType=cv2.LINE_AA)

    # left:
    if rng.uniform(0, 1) < p:
        alpha = random_angle()
        d = rng.uniform(0, w * size)
        x1 = - np.tan(alpha) * w / 2 + d
        x2 = + np.tan(alpha) * w / 2 + d
        x3 = min(0, x1)
        x4 = min(0, x2)
        points = np.array([[x1, 0], [x2, h], [x3, h], [x4, 0]], np.int32)
        cv2.fillPoly(im, [points], border_value, lineType=cv2.LINE_AA)

    # right:
    if rng.uniform(0, 1) < p:
        alpha = random_angle()
        d = rng.uniform(0, w * size)
        x1 = w - np.tan(alpha) * w / 2 - d
        x2 = w + np.tan(alpha) * w / 2 - d
        x3 = max(w, x1)
        x4 = max(w, x2)
        points = np.array([[x1, 0], [x2, h], [x3, h], [x4, 0]], np.int32)
        cv2.fillPoly(im, [points], border_value, lineType=cv2.LINE_AA)

    return im
