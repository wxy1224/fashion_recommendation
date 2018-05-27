import pandas as pd
import glob
import os
import shutil

PATH = 'train_modified.csv' # path of csv file
IMG_PATH = 'img/' # path of the img/ folder
DESTINATION = 'imgs/' # destination



df = pd.read_csv(PATH, usecols=['image_path']).as_matrix()

img_dict = {}

for i in range(len(df)):
    img_dict[df[i, 0]] = 1


img_list = glob.glob(IMG_PATH + '*/*.jpg')


count = 0
for img in img_list:
    if img not in img_dict:
        dest = os.path.join(DESTINATION, img)

        img_path = img.split('/')

        if not os.path.exists(os.path.join(DESTINATION, img_path[0])):
            os.makedirs(os.path.join(DESTINATION, img_path[0]))

        if not os.path.exists(os.path.join(DESTINATION, img_path[0], img_path[1])):
            os.makedirs(os.path.join(DESTINATION, img_path[0], img_path[1]))


        shutil.move(src=img, dst=dest)
        count += 1


