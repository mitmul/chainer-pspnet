#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainercv.transforms import resize_contain


def pre_img(img, crop_size):
    row, col = img.shape[1:]
    if row < crop_size:
        img = resize_contain(img, (crop_size, col))
    if col < crop_size:
        img = resize_contain(img, (row, crop_size))
    return img


def scale_process(img, base_size, scale):
    h, w = img.shape[1:]
