import random
import numpy as np
import cv2
from utils import *


def get_dataset_map():

    path = 'D:\data\wider_face\wider_face_train_bbx_gt.txt'
    lines = open(path, 'r')

    c = 1
    img_box_mapping = {}
    line = True
    while line:

        try:
            line = next(lines)
        except StopIteration:
            return img_box_mapping

        if '--' in line:
            img_name = line.replace('\n', '')
            number_of_boxes = int(next(lines).replace('\n', ''))

            img_boxes = []
            for _ in range(number_of_boxes):
                line = next(lines).replace('\n', '')
                box = line.split(' ')[:4]
                box = [int(i) for i in box]
                img_boxes.append(box)
            img_box_mapping[img_name] = img_boxes

        c += 1
        if c % 10 == 0:
            print(c)


def data_generator(input_size=(640, 640), batch_size=2):
    """
    bounding box:
    0 top left horizontal
    1 top left verical
    2 offset width
    3 offset hight
    """

    path = 'D:\data\wider_face\WIDER_train\images/'
    img_box_mapping = get_dataset_map()
    image_paths = list(img_box_mapping.keys())
    random.shuffle(image_paths)

    while True:
        y_batch = []
        img_batch = []
        for image_path in image_paths:
            img = cv2.imread(path + image_path)
            h, w, _ = img.shape
            y = np.zeros(shape=(h, w))
            for box in img_box_mapping[image_path]:  # box: [x, y, w, h]
                center_point = int(box[0] + box[2]/2), int(box[1] + box[3]/2)
                radius = ((box[2]/w + box[3]/h)*(h+w))/3
                y = draw_umich_gaussian(y, center_point, radius=int(radius))

            # resize
            img = cv2.resize(img, input_size)
            y = cv2.resize(y, input_size)

            # plot
            # overlay = np.tile(y[:, :, np.newaxis], [1, 1, 3]) * 500
            # combined = cv2.addWeighted(img.astype('uint8'), 0.4, overlay.astype('uint8'), 0.3, 1)
            # quick_plot(combined)

            img_batch.append(img)
            y_batch.append(y)

            if len(img_batch) == batch_size:
                yield np.stack(img_batch, axis=0), np.stack(y_batch, axis=0)
                img_batch = []
                y_batch = []


# test
# generator = data_generator(input_size=(640, 640), batch_size=5)
# for _ in range(10):
#     x, y = next(generator)
#     print(x.shape, y.shape)
