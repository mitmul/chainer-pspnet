#!/bin/bash

if [ ! -f 21.jpg ]; then
    curl -L -O http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21.jpg
fi
