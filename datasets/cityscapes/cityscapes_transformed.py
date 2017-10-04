from functools import partial

import cv2 as cv
import numpy as np
from PIL import Image

from chainer import datasets
from chainercv import transforms
from datasets.cityscapes.cityscapes_semantic_segmentation_dataset import \
    CityscapesSemanticSegmentationDataset


def _transform(inputs, mean=None, crop_size=(512, 512), color_sigma=25.5,
               scale=[0.5, 2.0], rotate=False, fliplr=False, n_class=20):
    img, label = inputs

    # Scaling
    if scale:
        if isinstance(scale, (list, tuple)):
            scale = np.random.uniform(scale[0], scale[1])
        scaled_h = int(img.shape[1] * scale)
        scaled_w = int(img.shape[2] * scale)
        img = transforms.resize(img, (scaled_h, scaled_w), Image.BICUBIC)
        label = transforms.resize(
            label[None, ...], (scaled_h, scaled_w), Image.NEAREST)[0]

    # Crop
    if crop_size is not None:
        if (img.shape[1] < crop_size[0]) or (img.shape[2] < crop_size[1]):
            shorter_side = min(img.shape[1:])
            _crop_size = (shorter_side, shorter_side)
            img, param = transforms.random_crop(img, _crop_size, True)
        else:
            img, param = transforms.random_crop(img, crop_size, True)
        label = label[param['y_slice'], param['x_slice']]

    # Rotate
    if rotate:
        angle = np.random.uniform(-10, 10)
        rows, cols = img.shape[1:]

        img = img.transpose(1, 2, 0)
        r = cv.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        img = cv.warpAffine(img, r, (cols, rows)).transpose(2, 0, 1)

        r = cv.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        label = cv.warpAffine(label, r, (cols, rows), flags=cv.INTER_NEAREST,
                              borderValue=-1)

    # Resize
    if crop_size is not None:
        if (img.shape[1] < crop_size[0]) or (img.shape[2] < crop_size[1]):
            img = transforms.resize(img, crop_size, Image.BICUBIC)
        if (label.shape[0] < crop_size[0]) or (label.shape[1] < crop_size[1]):
            label = transforms.resize(
                label[None, ...].astype(np.float32), crop_size, Image.NEAREST)
            label = label.astype(np.int32)[0]

    # Mean subtraction
    if mean is not None:
        img -= mean[:, None, None]

    # LR-flipping
    if fliplr:
        if np.random.rand() > 0.5:
            img = transforms.flip(img, x_flip=True)
            label = transforms.flip(label[None, ...], x_flip=True)[0]

    # Color augmentation
    if color_sigma is not None:
        img = transforms.pca_lighting(img, color_sigma)

    assert label.max() < n_class, '{}'.format(label.max())
    if crop_size is not None:
        assert img.shape == (3, crop_size[0], crop_size[1]), \
            '{} != {}'.format(img.shape, crop_size)
        assert label.shape == (crop_size[0], crop_size[1]), \
            '{} != {}'.format(label.shape, crop_size)

    return img, label


class CityscapesTransformedDataset(datasets.TransformDataset):

    # Cityscapes mean
    MEAN = np.array([73.15835921, 82.90891754, 72.39239876])

    def __init__(self, data_dir, label_resolution, split, ignore_labels=True,
                 crop_size=(713, 713), color_sigma=None, scale=[0.5, 2.0],
                 rotate=False, fliplr=False, n_class=19):
        self.d = CityscapesSemanticSegmentationDataset(
            data_dir, label_resolution, split)
        t = partial(
            _transform, mean=self.MEAN, crop_size=crop_size,
            color_sigma=color_sigma, scale=scale, rotate=rotate, fliplr=fliplr,
            n_class=n_class)
        super().__init__(self.d, t)
