# fashion_recommendation
CS231N Spring 2018 Final Project

This repository contains the code used to train and test our models.

The components of this repo include:
- deep-shopping-baseline: contains the code of our baseline, a 5-layer CNN model
- deep_fashion_to_coco.py: preprocess our fashion data into coco format so that they could be used in training our model
- tf-faster-rcnn folder: the faster rcnn model built in tensorflow and the main code structure is inherited from https://github.com/endernewton/tf-faster-rcnn
	- our experiment scripts can be found in experiments/
	- the model scripts can be found in lib/
- detectron_faster_rcnn_params: contains parameter files used to train the models
- Visualization.ipynb: notebook to visualize our loss and accuracy curves
- knn: the k-nearest neighbor code used to generate fashion recommendations based on a query image
