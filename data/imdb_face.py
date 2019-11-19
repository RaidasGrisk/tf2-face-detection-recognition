from utils import *
import pandas as pd
import urllib
import os
import tensorflow as tf


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)


def download_dataset():

    # init
    path = 'D:/data/imdb_face/IMDb-Face.csv'
    save_images_path = 'D:/data/imdb_face/images/'
    save_face_path = 'D:/data/imdb_face/faces/'
    df = pd.read_csv(path)

    # start from beginning or continue downloading
    finished_files = os.listdir('D:/data/imdb_face/faces/')
    finished_files = [int(i.split('.')[0]) for i in finished_files]

    if 'checked' not in df.columns:
        df['checked'] = ''

    c = 0
    for id, row in df.iterrows():

        print(id)
        if id in finished_files or row['checked'] in [1, 2, 3, 4]:
            continue

        # get image
        df.at[id, 'checked'] = 1
        try:
            response = urllib.request.urlopen(row['url']).read()
            img = cv2.imdecode(np.frombuffer(response, np.uint8), -1)
            df.at[id, 'checked'] = 2
        except urllib.error.HTTPError:
            df.at[id, 'checked'] = 3
            continue

        # next just in case img is None
        if img is None:
            df.at[id, 'checked'] = 4
            continue

        # check if image is RGB
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # get face box
        box = [int(i) for i in row['rect'].split(' ')]

        # check if resolutions match and fix face box if not
        h_, w_ = [int(i) for i in row['height width'].split(' ')]
        h, w, _ = img.shape
        if h != h_ or w != w_:
            h_fix = h / h_
            w_fix = w / w_
            box[0], box[2] = int(box[0] * h_fix), int(box[2] * h_fix)
            box[1], box[3] = int(box[1] * w_fix), int(box[3] * w_fix)

        face = img[box[1]:box[3], box[0]:box[2], :]

        # finish
        c += 1
        cv2.imwrite(save_images_path + str(id) + '.jpg', img)
        cv2.imwrite(save_face_path + str(id) + '.jpg', face)
        if c % 100 == 0:
            df.to_csv(path, index=True)


def data_generator(size=(96, 96), batch_size=1):

    # init
    path = 'D:/data/imdb_face/IMDb-Face.csv'
    save_face_path = 'D:/data/imdb_face/faces/'
    df = pd.read_csv(path)

    # delete bad files
    for i in os.listdir(save_face_path):
        file = save_face_path + i
        if os.path.getsize(file) < 1:
            print('removing {}'.format(i))
            os.remove(file)

    finished_files = os.listdir('D:/data/imdb_face/faces/')
    finished_files = [int(i.split('.')[0]) for i in finished_files]

    # filter people
    df = df[df.index.isin(finished_files)]
    df = df['name']

    batch = []
    while True:

        data_point = {}
        data_point['anchor'] = df.sample(1)
        data_point['positive'] = df[df.isin(data_point['anchor'])].sample(1)
        data_point['negative'] = df[~df.isin(data_point['anchor'])].sample(1)

        for i in data_point:
            img = cv2.imread(save_face_path + '/' + str(data_point[i].index[0]) + '.jpg')

            # augmentation
            img = tf.keras.preprocessing.image.random_rotation(img, 40, row_axis=0, col_axis=1, channel_axis=2)
            img = tf.keras.preprocessing.image.random_shear(img, 30, row_axis=0, col_axis=1, channel_axis=2)
            img = tf.keras.preprocessing.image.random_brightness(img, (0.2, 1.6))
            img = preprocess_recognition(img)
            data_point[i] = img

        data_point = np.stack([data_point['anchor'], data_point['positive'], data_point['negative']])
        batch.append(data_point)

        if len(batch) == batch_size:
            yield np.concatenate(batch, axis=0)
            batch = []


# gen = data_generator()
# x = gen.__next__()


def online_data_generator(input_size=(160, 160), batch_size=2):

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
