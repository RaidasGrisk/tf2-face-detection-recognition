"""
https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3
https://arxiv.org/pdf/1904.07850v2.pdf
https://github.com/xingyizhou/CenterNet

"""
import tensorflow as tf


class Backbone(tf.keras.Model):

    """
    Example of what is actually happening here.
    No loop for it makes it easier to understand.
    We are extracting 4 different layers from ResNet50/MobileNet.
    Dims will depend on the input size, but the relative
    sizes will stay the same and that is what matters.

    input       [1, 480, 640, 3]
    layer_1     [1, 15, 20, 2048]   (one of the last layers, 174th op in ResNet50)
    layer_2     [1, 30, 40, 1024]   (142)
    layer_3     [1, 60, 80, 512]    (80)
    layer_4     [1, 120, 160, 64]   (one of the first layers, 12th op in ResNet50)

    step 1: double the size of layer_1 -> [1, 30, 40, 2048]

    step 2: concat layer_1 [1, 30, 40, 2048] and layer_2 [1, 30, 40, 1024] -> [1, 30, 40, 3072]
            conv with stride 1 to -> [1, 30, 40, 128] this is just to decrease the num of filters (last dim)
            conv with stride 3 to -> [1, 30, 40, 128]
            double the size of this layer -> [1, 60, 80, 128]

    step 3: concat this [1, 60, 80, 128] and layer_2 [1, 60, 80, 512] -> [1, 60, 80, 640]
            conv to -> [1, 60, 80, 64]
            conv to -> [1, 60, 80, 64]
            resize  -> [1, 120, 160, 64]

    step 4: concat this [1, 120, 160, 64] and layer_4 [1, 120, 160, 64] -> [1, 120, 160, 128]
            conv to -> [1, 120, 160, 32]
            conv to -> [1, 120, 160, 32]
            conv to -> [1, 120, 160, 32] < -- these are the features we are going to be using in the model

    Last layer will serve us as the input for further branches of the model.

    """

    def __init__(self, backbone='resnet', input_shape=(640, 640, 3), training=True):
        super(Backbone, self).__init__()

        self.training = training
        self.backbone_name = backbone
        if backbone == 'mobilenet':
            self.baskbone = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape)
            self.layer_ids = [154, 119, 57, 30]
        else:
            self.baskbone = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape)
            self.layer_ids = [174, 142, 80, 12]

        self.baskbone.trainable = False
        self.backbone_layers = tf.keras.models.Model(
            inputs=self.baskbone.input,
            outputs=[self.baskbone.get_layer(index=i).output for i in self.layer_ids])

        self.l1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same')
        self.l2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same')
        self.l3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same')

        self.h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', use_bias=False)
        self.h2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)
        self.h3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)

        self.upsample1 = tf.keras.layers.UpSampling2D(size=[2, 2])  # upsampling does not have weights
        self.upsample2 = tf.keras.layers.UpSampling2D(size=[2, 2])
        self.upsample3 = tf.keras.layers.UpSampling2D(size=[2, 2])

        # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/BatchNormalization
        # Scale: if True, multiply by gamma. If False, gamma is not used.
        # When the next layer is linear (also e.g. nn.relu),
        # this can be disabled since the scaling will be done by the next layer.
        self.bn1 = tf.keras.layers.BatchNormalization(trainable=training)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=training)
        self.bn3 = tf.keras.layers.BatchNormalization(trainable=training)

        self.g1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

    def __call__(self, input):

        # layers extracted from Backbone model (ResNet or MobileNet):
        # 1st is the farthest one (near the end of the net),
        # 4th is the closest one (near the beggining)

        # preprocess input
        if 'mobile' in self.backbone_name.lower():
            input = tf.keras.applications.mobilenet.preprocess_input(input)
        else:
            input = tf.keras.applications.resnet50.preprocess_input(input)

        layer_1, layer_2, layer_3, layer_4 = self.backbone_layers(input)

        # step 1
        # layer_1 -> layer_1
        layer_1 = self.upsample1(layer_1)

        # step 2
        # layer_1 + layer_2 -> layer_12
        layer_12_conc = self.l1(tf.concat([layer_1, layer_2], axis=-1))
        layer_12_conv = self.h1(layer_12_conc)
        layer_12_conv = self.bn1(layer_12_conv)
        layer_12_conv = tf.nn.relu(layer_12_conv)
        layer_12 = self.upsample2(layer_12_conv)

        # step 3
        # layer_12 + layer_3 -> layer_123
        layer_123_conc = self.l2(tf.concat([layer_12, layer_3], axis=-1))
        layer_123_conv = self.h2(layer_123_conc)
        layer_123_conv = self.bn2(layer_123_conv)
        layer_123_conv = tf.nn.relu(layer_123_conv)
        layer_123 = self.upsample3(layer_123_conv)

        # step 4
        # layer_123 + layer_4 -> layer_1234
        layer_1234_conc = self.l3(tf.concat([layer_123, layer_4], axis=-1))
        layer_1234_conv = self.h3(layer_1234_conc)
        layer_1234_conv = self.bn3(layer_1234_conv)
        layer_1234_conv = tf.nn.relu(layer_1234_conv)

        layer_1234 = self.g1(layer_1234_conv)

        return layer_1234


class Detection(tf.keras.Model):

    def __init__(self):
        super(Detection, self).__init__()

        self.y_ = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation=tf.nn.sigmoid)
        self.upsample = tf.keras.layers.UpSampling2D(size=[4, 4])

    def __call__(self, input):

        y_ = self.y_(input)
        y_ = self.upsample(y_)
        y_ = tf.squeeze(y_, axis=-1)  # remove last axis
        return y_

    @staticmethod
    def loss(y, y_):
        loss = tf.keras.losses.binary_crossentropy(y, y_, from_logits=False, label_smoothing=0)
        return tf.reduce_mean(loss)

# https://missinglink.ai/guides/tensorflow/tensorflow-face-recognition-three-quick-tutorials/

# import GPUtil as GPU
#
# print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f} | Total {3:.0f}MB".format(
#     GPU.getGPUs()[0].memoryFree,
#     GPU.getGPUs()[0].memoryUsed,
#     GPU.getGPUs()[0].memoryUtil*100,
#     GPU.getGPUs()[0].memoryTotal))

