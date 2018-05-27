#!/bin/sh
python ../deep_fashion_to_coco.py
rm -rf data/cache
rm -rf output/vgg16
./experiments/scripts/train_faster_rcnn.sh 0 coco vgg16
./experiments/scripts/test_faster_rcnn.sh 0 coco vgg16
mv output/vgg16 output/vgg16_prev