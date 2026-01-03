import functools

import boxlib
import cameralib
import cv2
import numpy as np
import rlemasklib
import scipy.sparse as sps
from PIL import Image, ImageDraw
from simplepyutils import FLAGS
from sklearn import linear_model

import nlf.common.augmentation.background as bgaug
from nlf.common.util import TRAIN


@functools.lru_cache(512)
def load_array(path):
    return np.load(path, mmap_mode='r')


class MemMappedCSR:
    def __init__(self, path):
        self.path = path
        self.indices = load_array(path + '.indices.npy')
        self.indptr = load_array(path + '.indptr.npy')
        self.data = load_array(path + '.data.npy')
        self.shape = load_array(path + '.shape.npy')

    def __getitem__(self, row_slice):
        if not isinstance(row_slice, slice) or (
            row_slice.step != 1 and row_slice.step is not None
        ):
            raise NotImplementedError(
                'MemMappedCSR must be indexed with a single slice with step 1.'
            )

        data_slice = slice(self.indptr[row_slice.start], self.indptr[row_slice.stop])
        data = self.data[data_slice]
        indices = self.indices[data_slice]
        indptr = self.indptr[row_slice.start : row_slice.stop + 1] - self.indptr[row_slice.start]
        shape = (row_slice.stop - row_slice.start, self.shape[1])
        return sps.csr_matrix((data, indices, indptr), shape=shape, copy=False)


def augment_background(
    ex, im, orig_cam, cam, imshape, learning_phase, antialias, interp, background_rng
):
    has_realistic_background = any(
        x in ex.image_path.lower()
        for x in [
            'sailvos',
            'agora',
            'spec-syn',
            'hspace',
            'bedlam',
            'egobody',
            'egohumans',
            'rich',
        ]
    )

    has_gray_background = any(
        x in ex.image_path.lower() for x in ['hi4d_rerender', 'dfaust_render', 'thuman2_render']
    )

    bg_aug_prob = (
        0.2
        if has_realistic_background
        else (1.0 if has_gray_background else FLAGS.background_aug_prob)
    )
    if (
        FLAGS.background_aug_prob
        and (learning_phase == TRAIN or FLAGS.test_aug)
        and background_rng.random() < bg_aug_prob
    ):
        fgmask = rlemasklib.decode(ex.mask)
        fgmask = cameralib.reproject_image(
            fgmask, orig_cam, cam, imshape, antialias_factor=antialias, interp=interp
        )
        im = bgaug.augment_background(im, fgmask, background_rng)
    return im


def look_at_box(orig_cam, box, imshape):
    # The homographic reprojection of a rectangle (bounding box) will not be another rectangle
    # Hence, instead we transform the side midpoints of the short sides of the box and
    # determine an appropriate zoom factor by taking the projected distance of these two points
    # and scaling that to the desired output image side length.
    center_point = boxlib.center(box)
    if box[2] < box[3]:
        # Tall box: take midpoints of top and bottom sides
        delta_y = np.array([0, box[3] / 2])
        sidepoints = center_point + np.stack([-delta_y, delta_y])
    else:
        # Wide box: take midpoints of left and right sides
        delta_x = np.array([box[2] / 2, 0])
        sidepoints = center_point + np.stack([-delta_x, delta_x])

    cam = orig_cam.copy()
    cam.turn_towards(target_image_point=center_point)
    cam.undistort()
    cam.square_pixels()
    cam_sidepoints = cameralib.reproject_image_points(sidepoints, orig_cam, cam)
    crop_side = np.linalg.norm(cam_sidepoints[0] - cam_sidepoints[1])
    cam.zoom(FLAGS.proc_side / crop_side)
    cam.center_principal_point(imshape)
    return cam


def sparse_matrix_to_dict(sparse_matrix):
    return dict(
        indptr=np.int32(sparse_matrix.indptr),
        _ragged_data=np.float32(sparse_matrix.data),
        _ragged_indices=np.int32(sparse_matrix.indices),
        dense_shape=np.array(sparse_matrix.shape, np.int32),
    )


def recolor_border(im, border_value=(127, 127, 127)):
    is_valid_mask = np.any(im > 20, axis=-1)
    h, w = im.shape[:2]
    im_changed = im.copy()

    # bottom:
    last_valid_index_per_col = h - np.argmax(is_valid_mask[::-1], axis=0)
    is_any_valid_per_col = np.any(is_valid_mask, axis=0)
    last_valid_index_per_col[~is_any_valid_per_col] = 0

    col_inds = np.arange(w)
    quantile = 1e-1
    ransac_start = linear_model.QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
    ransac_end = linear_model.QuantileRegressor(quantile=1 - quantile, alpha=0, solver='highs')
    fitted = ransac_end.fit(col_inds[:, np.newaxis], last_valid_index_per_col)  # .estimator_
    offset = fitted.intercept_
    if offset < h - 1:
        offset -= 1

    slope = fitted.coef_[0]
    y1 = offset
    y2 = offset + slope * w
    y3 = max(h, y1)
    y4 = max(h, y2)
    points = np.array([[0, y1], [w, y2], [w, y3], [0, y4]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)

    # top:
    first_valid_index_per_col = np.argmax(is_valid_mask, axis=0)
    first_valid_index_per_col[~is_any_valid_per_col] = h
    fitted = ransac_start.fit(col_inds[:, np.newaxis], first_valid_index_per_col)  # .estimator_
    offset = fitted.intercept_
    if offset > 0:
        offset += 1

    slope = fitted.coef_[0]
    y1 = offset
    y2 = offset + slope * w
    y3 = min(0, y1)
    y4 = min(0, y2)
    points = np.array([[0, y1], [w, y2], [w, y3], [0, y4]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)

    # left:
    first_valid_index_per_row = np.argmax(is_valid_mask, axis=1)
    is_any_valid_per_row = np.any(is_valid_mask, axis=1)
    first_valid_index_per_row[~is_any_valid_per_row] = w
    row_inds = np.arange(h)
    fitted = ransac_start.fit(row_inds[:, np.newaxis], first_valid_index_per_row)  # .estimator_
    offset = fitted.intercept_
    if offset > 0:
        offset += 1
    slope = fitted.coef_[0]
    x1 = offset
    x2 = offset + slope * h
    x3 = min(0, x1)
    x4 = min(0, x2)
    points = np.array([[x1, 0], [x2, h], [x3, h], [x4, 0]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)

    # right:
    last_valid_index_per_row = w - np.argmax(is_valid_mask[:, ::-1], axis=1)
    last_valid_index_per_row[~is_any_valid_per_row] = 0
    fitted = ransac_end.fit(row_inds[:, np.newaxis], last_valid_index_per_row)  # .estimator_
    offset = fitted.intercept_
    if offset < w - 1:
        offset -= 1
    slope = fitted.coef_[0]
    x1 = offset
    x2 = offset + slope * h
    x3 = max(w, x1)
    x4 = max(w, x2)
    points = np.array([[x1, 0], [x2, h], [x3, h], [x4, 0]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)
    return im_changed


@functools.lru_cache()
def make_marker(size):
    s = size
    image = Image.new("RGB", (s, s), "red")
    draw = ImageDraw.Draw(image)

    # Draw red circle filling the image
    sm1 = s - 1
    draw.ellipse([0, 0, sm1, sm1], outline="blue")

    # Convert to numpy for OpenCV
    img_np = np.array(image)

    # Draw blue diagonals using OpenCV
    cv2.line(img_np, (0, 0), (sm1, sm1), (0, 0, 255), 2)
    cv2.line(img_np, (sm1, 0), (0, sm1), (0, 0, 255), 2)
    return img_np

@functools.lru_cache()
def make_marker_plus(size):
    s = size
    image = Image.new("RGB", (s, s), "green")
    draw = ImageDraw.Draw(image)
    sm1 = s - 1

    # Draw unfilled square along the border in yellow
    draw.rectangle([0, 0, sm1, sm1], outline="yellow")

    # Convert to numpy for OpenCV drawing
    img_np = np.array(image)

    # Compute center coordinates for even size: two central pixels
    centers = [s//2 - 1, s//2]

    # Draw plus sign manually for perfect center alignment
    green_bgr = (0, 255, 0)
    for y in centers:
        cv2.line(img_np, (0, y), (sm1, y), green_bgr, 1)
    for x in centers:
        cv2.line(img_np, (x, 0), (x, sm1), green_bgr, 1)

    return img_np
