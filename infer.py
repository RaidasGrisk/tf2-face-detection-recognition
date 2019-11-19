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

backbone.load_weights('/chechpoints/' + 'backbone')
detection.load_weights('/chechpoints/' + 'detection')
recognition.load_weights(cpkt_dir + 'recognition')

# load face vectors
faces, names = get_images_of_faces()
face_vectors = recognition(faces)

vid = cv2.VideoCapture(0)
while True:

    _, x = vid.read()
    # x = cv2.imread('D:/data/wider_face/WIDER_train/images/0--Parade/0_Parade_marchingband_1_842.jpg')
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

    boxes = heatmap_to_bounding_box(y_, max_thresh=0.8, mean_thresh=0.3)
    # recognize
    exctracted_faces = []
    recognized_names = []
    for box in boxes:
        img_face = x[:, box[1]:box[1] + box[3], box[0]:box[00] + box[2], :]
        img_face = preprocess_recognition(img_face[0, ::])[np.newaxis, ::]
        exctracted_faces.append(img_face)

    if len(exctracted_faces) > 0:
        face_vectors_ = recognition(np.concatenate(exctracted_faces, 0))
        for i in face_vectors_:
            distances = np.sum(np.square(face_vectors - i), axis=1)
            if all(distances > 0.25):
                name = 'unknown'
            else:
                person_id = np.argmin(distances)
                name = names[person_id]
                name = name + ' {:.2f}'.format(distances[person_id])
            recognized_names.append(name)

    for box, name in zip(boxes, recognized_names):
        combined = cv2.rectangle(combined, tuple(box[:2]), (box[0] + box[2], box[1] + box[3]), (0, 128, 0), 1)
        combined = cv2.putText(combined, name, tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)

    cv2.imshow('image', combined)
    # cv2.waitKey()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break

