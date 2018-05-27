from PIL import Image
import json
import os
import pandas as pd
import numpy as np
from random import shuffle


def prepare_category_dict(root_path="", is_training=True):
    if is_training:
        training_csv_path = os.path.join(root_path, 'train_modified.csv')
    else:
        training_csv_path = os.path.join(root_path, 'vali_modified.csv')
    print("reading csv file: ", training_csv_path)
    training_csv = pd.read_csv(training_csv_path, usecols=['image_path', 'category']).as_matrix()

    training_dict = {}

    for i in range(len(training_csv)):
        key = training_csv[i, 0]
        val = training_csv[i, 1]
        training_dict[key] = val

    return training_dict


def convert_to_coco_bbox(coors):
    x1 = float(coors[0])
    y1 = float(coors[1])
    x2 = float(coors[2])
    y2 = float(coors[3])
    W = x2 - x1
    H = y2 - y1
    return [x1, y1, W, H]

def get_synthetic_categories():
    categories = []
    for index in range(48):
        dic = {
            'id': index,
            'name': '{}'.format(index),
            'supercategory': 'fashion'
        }
        categories.append(dic)
    return categories

def get_categories(num_categories=5):
    complete_categories = []
    for i in range(num_categories+1):
        dic = {
            'id': i,
            'name': '{}'.format(i),
            'supercategory': 'fashion'
        }
        complete_categories.append(dic)
    return complete_categories

def dump_annotation_file(IS_TRAINING=True):
    SAVE_PATH = 'instances_fashion_train2018.json' if IS_TRAINING else 'instances_fashion_test2018.json'

    images, anns, categories = [], [], []
    images_to_annos = {}
    # put all your fashion data here img/Anno needs to be here.
    # root_path = 'tf-faster-rcnn/data/'
    coco_path = '/afs/cs.stanford.edu/u/xw1/fashion_recommendation/tf-faster-rcnn/data/coco/annotations/'
    # coco_path = '/home/feiliu/Desktop/cs231N_Spring_2018/final_project/tf-faster-rcnn/data/coco/annotations/'
    # root_path = '/afs/cs.stanford.edu/u/xw1/fashion_recommendation/tf-faster-rcnn/data/fashion/'
    # root_path = '/home/feiliu/Desktop/cs231N_Spring_2018/final_project/deep_fashion_data/'
    # docker_image = root_path #'/cs231_project/tf-faster-rcnn/data/deep_fashion_data/'
    # root_path = '/home/feiliu/Desktop/cs231N_Spring_2018/final_project/deep_fashion_data/'
    root_path = '/afs/cs.stanford.edu/u/xw1/fashion_recommendation/tf-faster-rcnn/data/fashion/'
    # docker_image = '/cs231_project/tf-faster-rcnn/data/deep_fashion_data/'
    # docker_image = '/afs/cs.stanford.edu/u/xw1/fashion_recommendation/tf-faster-rcnn/data/fashion/'
    docker_image = root_path
    category_file_path = root_path+"Anno/list_category_img.txt"
    json_output_path = os.path.join(coco_path, SAVE_PATH)
    # json_output_path = '/home/feiliu/Desktop/cs231N_Spring_2018/final_project/fashion_recommendation/tf-faster-rcnn/data/coco/annotations/instances_fashion_train2018.json'
    # json_output_path = '/afs/cs.stanford.edu/u/xw1/fashion_recommendation/tf-faster-rcnn/data/coco/annotations/instances_fashion_train2018.json'
    bbox_file_path = root_path+"Anno/list_bbox.txt"
    subsample_limit = 10 #600000000

    categorical_dict = prepare_category_dict(root_path, IS_TRAINING)


    with open(category_file_path, 'r') as f:
        content = f.readlines()
        content = content[2:]
    shuffle(content)
    bbox_map = {}

    with open(bbox_file_path, 'r') as f:
        bbox_content = f.readlines()
        bbox_content = bbox_content[2:]
        for bbox_line in bbox_content:
            pair = bbox_line.split()
            image_path = docker_image + pair[0]
            bbox_coors =pair[1:]
            bbox_map[image_path]=convert_to_coco_bbox(bbox_coors)

    i = 0
    category_map = {}
    category_set = set()
    for line in content:
        if i > subsample_limit:
            break
        pair = line.split()
        image_read_path = root_path + pair[0]
        image_path = docker_image + pair[0]

        
        
        img = Image.open(image_read_path)
        width, height = img.size

        if pair[0] in categorical_dict:
            category_map[image_path]=categorical_dict[pair[0]]
            dic = {'file_name': pair[0], 'id': image_path, 'height': height, 'width': width}
            images.append(dic)
            i += 1
            # categories.append(categorical_dict[pair[0]])
            category_set.add(categorical_dict[pair[0]])

        else:
            continue
    print("category set:", category_set)
    ann_index = 0

    for image_dic in images:
        image_path = image_dic['id']
        bbox_coors = bbox_map[image_path]
        dic2 = {'segmentation': [], 'area': bbox_coors[2]*bbox_coors[3],
                'iscrowd': 0, 'image_id': image_dic['id'], 'bbox': bbox_coors,
                'category_id': category_map[image_path], 'id': ann_index}
        ann_index+=1
        anns.append(dic2)


    assert len(images) == len(anns)
    # assert len(images) == len(categories)
    print(len(images))

    data = {'images':images, 'annotations':anns, 'categories':get_categories()}

    with open(json_output_path, 'w') as outfile:
        json.dump(data, outfile)

if __name__=='__main__':
    dump_annotation_file(True)
    dump_annotation_file(False)
    # prepare_category_dict(root_path="", is_training=True)

