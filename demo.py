#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import chainer
import matplotlib.pyplot as plot
from chainer import serializers
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_label

from datasets import cityscapes_label_colors
from datasets import cityscapes_label_names
from evaluate import inference
from evaluate import preprocess
from pspnet import PSPNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_fn', '-f', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument(
        '--single_scale', '-s', action='store_true', default=False)
    parser.add_argument(
        '--model', '-m', type=str, choices=['VOC', 'Cityscapes', 'ADE20K'])
    args = parser.parse_args()

    if args.model == 'VOC':
        n_class = 21
        n_blocks = [3, 4, 23, 3]
        feat_size = 60
        mid_stride = True
        param_fn = 'weights/pspnet101_VOC2012_473_reference.chainer'
        base_size = 512
        crop_size = 473
        labels = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors
    elif args.model == 'Cityscapes':
        n_class = 19
        n_blocks = [3, 4, 23, 3]
        feat_size = 90
        mid_stride = True
        param_fn = 'weights/pspnet101_cityscapes_713_reference.chainer'
        base_size = 2048
        crop_size = 713
        labels = cityscapes_label_names
        colors = cityscapes_label_colors
    elif args.model == 'ADE20K':
        n_class = 150
        n_blocks = [3, 4, 6, 3]
        feat_size = 60
        mid_stride = False
        param_fn = 'weights/pspnet101_ADE20K_473_reference.chainer'
        base_size = 512
        crop_size = 473

    chainer.config.train = False
    model = PSPNet(n_class, n_blocks, feat_size, mid_stride=mid_stride)
    serializers.load_npz(param_fn, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    img = preprocess(read_image(args.img_fn))

    # Inference
    pred = inference(
        model, n_class, base_size, crop_size, img, not args.single_scale)

    # Save the result image
    ax = vis_image(img)
    _, legend_handles = vis_label(pred, labels, colors, alpha=1.0, ax=ax)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.)
    base = os.path.splitext(os.path.basename(args.img_fn))[0]
    plot.savefig('predict_{}.png'.format(base), bbox_inches='tight', dpi=400)
