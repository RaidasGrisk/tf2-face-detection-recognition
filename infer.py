import tensorflow as tf
from models.detection import Backbone, Detection
from models.recognition import Recognizer
from utils import quick_plot, heatmap_to_bounding_box, get_images_of_faces, preprocess_recognition
import numpy as np
import cv2

# init
input_shape = (640, 640, 3)
backbone = Backbone(backbone='mobilenet', input_shape=input_shape)
detection = Detection()
recognition = Recognizer(training=False)
cpkt_dir = 'checkpoints/'
face_dir = '/faces/'

backbone.load_weights('/checkpoints/' + 'backbone')
detection.load_weights('/checkpoints/' + 'detection')
recognition.load_weights(cpkt_dir + 'recognition')

# load face vectors
faces, names = get_images_of_faces()
face_vectors = recognition(faces)

# main loop
vid = cv2.VideoCapture('data/other/vid.mp4')
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (1194, 570))
while vid.isOpened():

    _, x_raw = vid.read()
    # x_raw = cv2.imread('D:/data/wider_face/WIDER_train/images/4--Dancing/4_Dancing_Dancing_4_226.jpg')
    x = tf.expand_dims(tf.image.resize(x_raw, input_shape[:2]), 0)

    # detect
    sharedconv = backbone(x)
    y_ = detection(sharedconv)
    y_ = tf.image.resize(tf.expand_dims(y_, -1), x_raw.shape[:2])
    boxes = heatmap_to_bounding_box(y_, max_thresh=0.8, mean_thresh=0.4)

    # extract faces
    extracted_faces = []
    for box in boxes:
        img_face = x_raw[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :]
        # quick_plot(img_face)
        img_face = preprocess_recognition(img_face)
        # quick_plot((img_face*255))
        extracted_faces.append(img_face)

    # recognize
    recognized_faces = []
    if len(extracted_faces) > 0:
        face_vectors_ = recognition(tf.stack(extracted_faces))
        for face_, box in zip(face_vectors_, boxes):
            distances = tf.reduce_sum(tf.square(face_vectors - face_), axis=1)
            person_id = np.argmin(distances)
            name = [names[person_id] if all(distances > 0.15) else 'unknown'][0]
            face_prob = box[-1].round(2)
            min_dist = distances[person_id].numpy().round(2)
            recognized_faces.append([name, face_prob, min_dist])

    # combine y_ mask
    overlay = y_.numpy()[0, :, :]
    overlay[overlay < 0.5] = 0
    overlay = overlay * 500
    overlay = cv2.applyColorMap(overlay.astype('uint8'), cv2.COLORMAP_JET)
    overlay[(overlay[:, :, 0] == 128) & (overlay[:, :, 1] == 0)] = 0
    combined = cv2.addWeighted(x_raw, 0.8, overlay.astype('uint8'), 0.5, 0)

    # add boxes and related data
    for box, face in zip(boxes, recognized_faces):
        combined = cv2.rectangle(combined, tuple(box[:2]), (box[0] + box[2], box[1] + box[3]), (43, 255, 0), 1)
        coords_0 = box[0] + box[2], box[1] + 13
        coords_1 = coords_0[0], coords_0[1] + 13
        coords_2 = coords_1[0], coords_1[1] + 13
        combined = cv2.putText(combined, 'mask_avg: ' + str(face[1]), coords_0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (43, 255, 0), 1)
        combined = cv2.putText(combined, 'dist_min: ' + str(face[2]), coords_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (43, 255, 0), 1)
        combined = cv2.putText(combined, 'face_id: ' + face[0], coords_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (43, 255, 0), 1)

    # out.write(combined)
    cv2.imshow('image', combined)
    # cv2.waitKey()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        # out.release()
        break

