import tensorflow as tf
from model import Backbone, Detection
from data import data_generator
from utils import quick_plot, heatmap_to_bounding_box
import numpy as np
import cv2

# init
input_shape = (640, 640, 3)
backbone = Backbone(backbone='mobilenet', input_shape=input_shape)
detection = Detection()
cpkt_dir = '/chechpoints/'
backbone.load_weights(cpkt_dir + 'backbone')
detection.load_weights(cpkt_dir + 'detection')


vid = cv2.VideoCapture(0)
while True:

    _, x = vid.read()
    x = cv2.imread('D:/data/wider_face/WIDER_train/images/0--Parade/0_Parade_marchingband_1_842.jpg')
    x = cv2.resize(x, input_shape[:2])[np.newaxis, :, :, :]

    # forward-prop
    sharedconv = backbone(x)
    y_ = detection(sharedconv)

    # plot result
    overlay = y_.numpy()[0, :, :]
    overlay[overlay < 0.5] = 0
    overlay = overlay * 500
    overlay = cv2.applyColorMap(overlay.astype('uint8'), cv2.COLORMAP_JET)
    overlay[(overlay[:, :, 0] == 128) & (overlay[:, :, 1] == 0)] = 0
    combined = cv2.addWeighted(x.astype('uint8')[0, :, :, :], 0.4, overlay.astype('uint8'), 0.3, 1)

    boxes = heatmap_to_bounding_box(y_, max_thresh=0.9, mean_thresh=0.4)
    for box in boxes:
        combined = cv2.rectangle(combined, tuple(box[:2]), (box[0] + box[2], box[1] + box[3]), (0, 128, 0), 1)

    cv2.imshow('image', combined)
    # cv2.waitKey()
    # cv2.imshow('a', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break

