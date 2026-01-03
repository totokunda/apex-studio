import cv2
import numpy as np
import numba


def augment_color(im, rng, out_dtype=None):
    if out_dtype is None:
        out_dtype = im.dtype

    if im.dtype == np.uint8:
        im = cv2.divide(im, (255, 255, 255, 255), dtype=cv2.CV_32F)

    augmentation_functions = [
        augment_brightness,
        augment_contrast,
        augment_hue,
        augment_saturation,
    ]
    rng.shuffle(augmentation_functions)

    colorspace = 'rgb'
    for fn in augmentation_functions:
        colorspace = fn(im, colorspace, rng)

    if colorspace != 'rgb':
        cv2.cvtColor(im, cv2.COLOR_HSV2RGB, dst=im)

    np.clip(im, 0, 1, out=im)

    if out_dtype == np.uint8:
        return (im * 255).astype(np.uint8)
    else:
        return im


def augment_brightness(im, in_colorspace, rng):
    if in_colorspace != 'rgb':
        cv2.cvtColor(im, cv2.COLOR_HSV2RGB, dst=im)

    im += rng.uniform(-0.125, 0.125)
    return 'rgb'


def augment_contrast(im, in_colorspace, rng):
    if in_colorspace != 'rgb':
        cv2.cvtColor(im, cv2.COLOR_HSV2RGB, dst=im)
    # im -= 0.5
    # im *= rng.uniform(0.5, 1.5)
    # im += 0.5
    _augment_contrast_nb(im, rng.uniform(0.5, 1.5))
    return 'rgb'


@numba.njit(cache=True)
def _augment_contrast_nb(im, factor):
    im_flat = im.reshape(-1)
    factor32 = np.float32(factor)
    offset = np.float32(-0.5) * factor32 + np.float32(0.5)
    for i in range(im_flat.shape[0]):
        im_flat[i] = im_flat[i] * factor32 + offset


def augment_hue(im, in_colorspace, rng):
    if in_colorspace != 'hsv':
        np.clip(im, 0, 1, out=im)
        cv2.cvtColor(im, cv2.COLOR_RGB2HSV, dst=im)
    # hue = im[:, :, 0]
    # hue += rng.uniform(-72, 72)
    # hue[hue < 0] += 360
    # hue[hue > 360] -= 360
    _augment_hue_nb(im, rng.uniform(-72, 72))
    return 'hsv'


@numba.njit(cache=True)
def _augment_hue_nb(im, offset):
    im_flat = im.reshape(-1, 3)
    for i in range(im_flat.shape[0]):
        im_flat[i, 0] += offset
        if im_flat[i, 0] < 0:
            im_flat[i, 0] += 360
        elif im_flat[i, 0] > 360:
            im_flat[i, 0] -= 360


def augment_saturation(im, in_colorspace, rng):
    if in_colorspace != 'hsv':
        np.clip(im, 0, 1, out=im)
        cv2.cvtColor(im, cv2.COLOR_RGB2HSV, dst=im)

    # saturation = im[:, :, 1]
    # saturation *= rng.uniform(0.5, 1.5)
    # saturation[saturation > 1] = 1
    _augment_saturation_nb(im, rng.uniform(0.5, 1.5))
    return 'hsv'


@numba.njit(cache=True)
def _augment_saturation_nb(im, factor):
    im_flat = im.reshape(-1, 3)
    for i in range(im_flat.shape[0]):
        im_flat[i, 1] *= factor
        if im_flat[i, 1] > 1:
            im_flat[i, 1] = 1
