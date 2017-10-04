PSPNet
======

This is an unofficial implementation of Pyramid Scene Parsing Network (PSPNet) in [Chainer](https://github.com/chainer/chainer).

![](https://github.com/mitmul/chainer-pspnet/wiki/images/demoVideo.gif)

# Training

## Requirement

- Python 3.4.4+
    - Chainer 3.0.0b1+
    - ChainerMN master
    - CuPy 2.0.0b1+
    - ChainerCV 0.6.0+
    - NumPy 1.12.0+
    - tqdm 4.11.0+

```
pip install chainer --pre
pip install cupy --pre
pip install git+git://github.com/chainer/chainermn
pip install git+git://github.com/chainer/chainercv
pip install tqdm
````

---

# Inference using converted weights

## Requirement

- Python 3.4.4+
    - Chainer 3.0.0b1+
    - ChainerCV 0.6.0+
    - Matplotlib 2.0.0+
    - CuPy 2.0.0b1+
    - tqdm 4.11.0+

## 1. Run demo.py

### Cityscapes

```
$ python demo.py -g 0 -m cityscapes -f aachen_000000_000019_leftImg8bit.png
```

### Pascal VOC2012

```
$ python demo.py -g 0 -m voc2012 -f 2008_000005.jpg
```

### ADE20K

```
$ python demo.py -g 0 -m ade20k -f ADE_val_00000001.jpg
```

### FAQ

If you get `RuntimeError: Invalid DISPLAY variable`, how about specifying the matplotlib's backend by an environment variable?

```
$ MPLBACKEND=Agg python demo.py -g 0 -m cityscapes -f aachen_000000_000019_leftImg8bit.png
```

---

# Convert weights by yourself

**Caffe is NOT needed** to convert `.caffemodel` to Chainer model. Use `caffe_pb2.py`.

## Requirement

- Python 3.4.4+
    - protobuf 3.2.0+
    - Chainer 3.0.0b1+
    - NumPy 1.12.0+

## 1. Download the original weights

Please download the weights below from the author's repository:

- pspnet50\_ADE20K.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCN1R3QnUwQ0hoMTA)
- pspnet101\_VOC2012.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCNVhETE5vVUdMYk0)
- pspnet101\_cityscapes.caffemodel: [GoogleDrive](https://drive.google.com/open?id=0BzaU285cX7TCT1M3TmNfNjlUeEU)

**and then put them into `weights` directory.**

## 2. Convert weights

```
$ python convert.py
```

---

# Reference

- The original implementation by authors is: [hszhao/PSPNet](https://github.com/hszhao/PSPNet)
- The original paper is:
    - Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia, "Pyramid Scene Parsing Network", Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
