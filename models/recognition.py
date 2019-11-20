"""
https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3
https://arxiv.org/pdf/1904.07850v2.pdf
https://github.com/xingyizhou/CenterNet

"""
import tensorflow as tf


class Recognizer(tf.keras.Model):

    def __init__(self, training, output_dim=128):
        super(Recognizer, self).__init__()

        self.conv_params1 = {'filters': 3, 'kernel_size': 3, 'padding': 'valid', 'activation': None}
        self.conv_params2 = {'filters': 9, 'kernel_size': 3, 'padding': 'valid', 'activation': None}
        self.conv_params3 = {'filters': 16, 'kernel_size': 3, 'padding': 'valid', 'activation': None}
        self.conv_params4 = {'filters': 32, 'kernel_size': 3, 'padding': 'valid', 'activation': None}
        self.pool_params = {'pool_size': 2, 'padding': 'valid'}

        self.l1 = tf.keras.layers.Conv2D(**self.conv_params1)
        self.bn1 = tf.keras.layers.BatchNormalization(trainable=training)
        self.l2 = tf.keras.layers.Conv2D(**self.conv_params1)
        self.l3 = tf.keras.layers.AveragePooling2D(**self.pool_params)

        self.l4 = tf.keras.layers.Conv2D(**self.conv_params2)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=training)
        self.l5 = tf.keras.layers.Conv2D(**self.conv_params2)
        self.l6 = tf.keras.layers.AveragePooling2D(**self.pool_params)

        self.l7 = tf.keras.layers.Conv2D(**self.conv_params3)
        self.bn3 = tf.keras.layers.BatchNormalization(trainable=training)
        self.l8 = tf.keras.layers.Conv2D(**self.conv_params3)
        self.l9 = tf.keras.layers.AveragePooling2D(**self.pool_params)

        self.l10 = tf.keras.layers.Conv2D(**self.conv_params4)
        # self.l11 = tf.keras.layers.AveragePooling2D(**self.pool_params)
        self.l12 = tf.keras.layers.Conv2D(**self.conv_params4)

        self.flatten = tf.keras.layers.Flatten()
        self.final = tf.keras.layers.Dense(output_dim)

    def call(self, y_):

        # print(y_.shape)
        y_ = self.l1(y_)
        y_ = self.bn1(y_)
        y_ = tf.nn.leaky_relu(y_)
        y_ = self.l2(y_)
        y_ = tf.nn.leaky_relu(y_)
        y_ = self.l3(y_)
        # print(y_.shape)

        y_ = self.l4(y_)
        y_ = self.bn2(y_)
        y_ = tf.nn.leaky_relu(y_)
        y_ = self.l5(y_)
        y_ = tf.nn.leaky_relu(y_)
        y_ = self.l6(y_)
        # print(y_.shape)

        y_ = self.l7(y_)
        y_ = self.bn3(y_)
        y_ = tf.nn.leaky_relu(y_)
        y_ = self.l8(y_)
        y_ = tf.nn.leaky_relu(y_)
        y_ = self.l9(y_)
        # print(y_.shape)

        y_ = self.l10(y_)
        y_ = tf.nn.leaky_relu(y_)
        # y_ = self.l11(y_)
        y_ = self.l12(y_)
        y_ = tf.nn.leaky_relu(y_)
        # print(y_.shape)

        # y_ = tf.squeeze(y_, axis=-1)
        y_ = self.flatten(y_)
        # print(y_.shape)
        y_ = self.final(y_)
        # print(y_.shape)

        return y_

    def loss(self, y_, margin=0.2):

        # gather indices for anchor, positive and negative samples
        # the input shape is [B, 96, 96, 3] where B is number of images
        # B is ordered like this: [a, p, n, a, p, n ... a, p, n]
        # reorder y_ so that there are 3 tensors for anchor, pos and neg

        data_points, _ = y_.shape
        anchor_idx = list(range(data_points))[::3]
        pos_idx = [i + 1 for i in anchor_idx]
        neg_idx = [i + 2 for i in anchor_idx]
        anchor_idx, pos_idx, neg_idx = [[i] for i in anchor_idx], [[i] for i in pos_idx], [[i] for i in neg_idx]
        anchor, pos, neg = tf.gather_nd(y_, anchor_idx), tf.gather_nd(y_, pos_idx), tf.gather_nd(y_, neg_idx),

        # let's minimize:
        # distance from anchor to pos (should be small) -
        # distance from anchor to neg (should be big)
        # add margin, so that just setting all 0 does not work
        # basically try to maximize the distance between pos and neg dist
        pos_dist = tf.reduce_sum(tf.square(anchor - pos), axis=-1)
        neg_dist = tf.reduce_sum(tf.square((anchor - neg)), axis=-1)
        basic_loss = (pos_dist - neg_dist) + margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return loss

#
# model = Recognizer(training=True)
# model.build(input_shape=(3, 96, 96, 3))
# model.summary()
# y_ = model(tf.random.uniform((3, 96, 96, 3)))
# print(y_.shape)
