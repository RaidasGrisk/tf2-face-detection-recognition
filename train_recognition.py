import tensorflow as tf
from models.recognition import Recognizer
from data.imdb_face import data_generator

cpkt_dir = 'checkpoints/'
save_iter = 100
data_gen = data_generator(batch_size=64)
model = Recognizer(training=True)
model.load_weights(cpkt_dir + 'recognition')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss_hist = []
for x in data_gen:

    with tf.GradientTape() as tape:
        y_ = model(x)
        loss = model.loss(y_, margin=0.2)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(loss.numpy())

    if loss.numpy() != 0:
        loss_hist.append(loss.numpy())
    if len(loss_hist) % save_iter == 0:
        with open(cpkt_dir + 'loss.txt', 'a') as file:
            [file.write(str(i) + '\n') for i in loss_hist]
        loss_hist = []
        model.save_weights(cpkt_dir + 'recognition')