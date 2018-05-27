#!/bin/sh
python ../deep_fashion_to_coco.py
rm -rf data/cache
./experiments/scripts/train_faster_rcnn.sh 0 coco vgg16
./experiments/scripts/test_faster_rcnn.sh 0 coco vgg16