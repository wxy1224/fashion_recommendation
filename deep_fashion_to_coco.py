

from PIL import Image
import json
def convert_to_coco_bbox(coors):
    x1 = float(coors[0])
    y1 = float(coors[1])
    x2 = float(coors[2])
    y2 = float(coors[3])
    W = x2 - x1
    H = y2 - y1
    return [x1, y1, W, H]

def get_categories():
    categories = []
    for index in range(48):
        dic = {
            'id': index,
            'name': '{}'.format(index),
            'supercategory': 'fashion'
        }
        categories.append(dic)
    return categories

if __name__=='__main__':
    images, anns = [], []
    images_to_annos = {}
    # put all your fashion data here img/Anno needs to be here.
    root_path = '/home/feiliu/Desktop/cs231N_Spring_2018/final_project/deep_fashion_data/'
    category_file_path = root_path+"Anno/list_category_img.txt"
    json_output_path = '/home/feiliu/Desktop/cs231N_Spring_2018/final_project/tf-faster-rcnn/data/instances_minival2014.json'
    bbox_file_path = root_path+"Anno/list_bbox.txt"
    with open(category_file_path, 'r') as f:
        content = f.readlines()
        content = content[2:]
    bbox_map = {}

    with open(bbox_file_path, 'r') as f:
        bbox_content = f.readlines()
        bbox_content = bbox_content[2:]
        for bbox_line in bbox_content:
            pair = bbox_line.split()
            image_path = root_path + pair[0]
            bbox_coors =pair[1:]
            bbox_map[image_path]=convert_to_coco_bbox(bbox_coors)

    i = 0
    category_map = {}
    for line in content:
        pair = line.split()
        image_path = root_path + pair[0]
        category_id = int(pair[1])
        category_map[image_path]=category_id
        img = Image.open(image_path)
        width, height = img.size
        dic = {'file_name': image_path, 'id': i, 'height': height, 'width': width}
        images.append(dic)
        i += 1

    ann_index = 0

    for image_dic in images:
        image_path = image_dic['file_name']
        bbox_coors = bbox_map[image_path]
        dic2 = {'segmentation': [], 'area': bbox_coors[2]*bbox_coors[3],
                'iscrowd': 0, 'image_id': image_dic['id'], 'bbox': bbox_coors,
                'category_id': category_map[image_path], 'id': ann_index}
        ann_index+=1
        anns.append(dic2)

    data = {'images':images, 'annotations':anns, 'categories':get_categories()}

    with open(json_output_path, 'w') as outfile:
        json.dump(data, outfile)
