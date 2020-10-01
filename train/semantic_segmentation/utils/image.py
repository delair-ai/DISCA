import itertools

import buzzard as buzz
import cv2 as cv
import numpy as np
import torch


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[1], step):
        if x + window_size[0] > top.shape[1]:
            x = top.shape[1] - window_size[0]
        for y in range(0, top.shape[2], step):
            if y + window_size[1] > top.shape[2]:
                y = top.shape[2] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def from_coord_to_patch(img, coords, device):
    """Returns patches of the input image. coors is an output of grouper(n, sliding window(...))"""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    image_patches = [img[:, x : x + w, y : y + h] for x, y, w, h in coords]
    image_patches = torch.stack(image_patches).to(device)
    return image_patches
