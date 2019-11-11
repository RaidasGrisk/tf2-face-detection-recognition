import random
import numpy as np
import cv2
from utils import *
import pandas as pd

import urllib
from io import BytesIO

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)


def data_generator(input_size=(160, 160), batch_size=2):

    # init
    path = 'D:/data/imdb_face/IMDb-Face.csv'
    df = pd.read_csv(path)
    names = df['name'].unique()

    for _ in range(10):

        # choose triplet ids
        name = np.random.choice(names)
        triplet_ids = df[df['name'].isin([name])].sample(2).index.tolist() \
                  + df[~df['name'].isin([name])].sample(1).index.tolist()

        # check if urls are valid
        try:
            urllib.request.urlopen(df[df.index.isin([triplet_ids[0]])]['url'].values[0])
            urllib.request.urlopen(df[df.index.isin([triplet_ids[1]])]['url'].values[0])
            urllib.request.urlopen(df[df.index.isin([triplet_ids[2]])]['url'].values[0])
        except urllib.error.HTTPError:
            print('HTTP Error 404')
            continue

        for id in triplet_ids:
            url = df[df.index.isin([id])]['url'].values[0]
            response = urllib.request.urlopen(url).read()
            img = cv2.imdecode(np.frombuffer(response, np.uint8), -1)
            box = [int(i) for i in df[df.index.isin([id])]['rect'].values[0].split(' ')]

            # check if image is RGB
            if len(img.shape) != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # check if resolutions match and fix if not
            h_, w_ = [int(i) for i in df[df.index.isin([id])][['height width']].values[0][0].split(' ')]
            h, w, _ = img.shape
            if h != h_ or w != w_:
                print(h, h_, w, w_)
                h_fix = h / h_
                w_fix = w / w_
                box[0], box[2] = int(box[0] * h_fix), int(box[2] * h_fix)
                box[1], box[3] = int(box[1] * w_fix), int(box[3] * w_fix)

            img_cropped = img[box[1]:box[3], box[0]:box[2], :]
            img_cropped_resized = cv2.resize(img_cropped, input_size)
            quick_plot(img_cropped_resized)

#
#
# def data_generator(input_size=(640, 640), batch_size=2):
#     """
#     bounding box:
#     0 top left horizontal
#     1 top left verical
#     2 offset width
#     3 offset hight
#     """
#
#     path = 'D:\data\wider_face\WIDER_train\images/'
#     img_box_mapping = get_dataset_map()
#     image_paths = list(img_box_mapping.keys())
#     random.shuffle(image_paths)
#
#     while True:
#         y_batch = []
#         img_batch = []
#         for image_path in image_paths:
#             img = cv2.imread(path + image_path)
#             h, w, _ = img.shape
#             y = np.zeros(shape=(h, w))
#             for box in img_box_mapping[image_path]:  # box: [x, y, w, h]
#                 center_point = int(box[0] + box[2]/2), int(box[1] + box[3]/2)
#                 radius = ((box[2]/w + box[3]/h)*(h+w))/3
#                 y = draw_umich_gaussian(y, center_point, radius=int(radius))
#
#             # resize
#             img = cv2.resize(img, input_size)
#             y = cv2.resize(y, input_size)
#
#             # plot
#             overlay = np.tile(y[:, :, np.newaxis], [1, 1, 3]) * 500
#             combined = cv2.addWeighted(img.astype('uint8'), 0.4, overlay.astype('uint8'), 0.3, 1)
#             quick_plot(combined)
#
#             img_batch.append(img)
#             y_batch.append(y)
#
#             if len(img_batch) == batch_size:
#                 yield np.stack(img_batch, axis=0), np.stack(y_batch, axis=0)
#                 img_batch = []
#                 y_batch = []
#

# test
# generator = data_generator(input_size=(640, 640), batch_size=5)
# for _ in range(10):
#     x, y = next(generator)
#     print(x.shape, y.shape)
