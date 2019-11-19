import tensorflow as tf
from models.detection import Backbone, Detection
from data.wider_face import data_generator
from utils import quick_plot
import numpy as np
import cv2

# init
input_shape = (640, 640, 3)
backbone = Backbone(backbone='mobilenet', input_shape=input_shape)
detection = Detection()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
data_gen = data_generator(input_size=input_shape[:2], batch_size=1)
cpkt_dir = 'checkpoints/'
backbone.load_weights(cpkt_dir + 'backbone')
detection.load_weights(cpkt_dir + 'detection')
c = 0
save_iter = 10
loss_hist = []

for x, y in data_gen:

    c += 1
    with tf.GradientTape() as tape:

        # forward-prop
        sharedconv = backbone(x)
        y_ = detection(sharedconv)

        # loss
        loss = detection.loss(y, y_)
        # loss = tf.reduce_mean(loss)

    # backward-prop
    model_variables = backbone.trainable_variables + detection.trainable_variables

    grads = tape.gradient(loss, model_variables)
    optimizer.apply_gradients(zip(grads, model_variables))

    loss_hist.append(loss.numpy().mean())
    print(c, loss.numpy().mean())

    # plot result
    if c % 20 == 0:
        overlay = np.tile(y_[0, :, :, np.newaxis], [1, 1, 3]) * 500
        combined = cv2.addWeighted(x.astype('uint8')[0, :, :, :], 0.4, overlay.astype('uint8'), 0.3, 1)
        quick_plot(combined)

    # save
    if c % save_iter == 0:
        backbone.save_weights(cpkt_dir + 'backbone')
        detection.save_weights(cpkt_dir + 'detection')

        with open('loss.txt', 'w') as file:
            [file.write(str(s) + '\n') for s in loss_hist]

# plot result
overlay = np.tile(y_[0, :, :, np.newaxis], [1, 1, 3]) * 500
combined = cv2.addWeighted(x.astype('uint8')[0, :, :, :], 0.4, overlay.astype('uint8'), 0.3, 1)
quick_plot(combined)

# https://github.com/tensorflow/community/blob/master/rfcs/20181016-replicator.md
# grads1 = [i.numpy() for i in grads]
# [(i - j).mean() for i, j in zip(grads0, grads1)]