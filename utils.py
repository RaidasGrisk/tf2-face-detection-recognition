import numpy as np
import cv2
import os


def quick_plot(x):
    cv2.imshow('org', x.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def heatmap_to_bounding_box(y_, max_thresh, mean_thresh):
    # Grayscale then Otsu's threshold
    # https://stackoverflow.com/questions/58419893/generating-bounding-boxes-from-heatmap-data
    pred = y_.numpy()[0, :, :] * 2
    thresh = cv2.threshold(pred.astype('uint8'), 1, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        heat_area = y_.numpy()[0, y:y+h, x:x+w]
        max_score, mean_score = heat_area.max(), heat_area.mean()
        if max_score > max_thresh and mean_score > mean_thresh:
            boxes.append([x, y, w, h, max_score, mean_score])
    return boxes


def gaussian_radius(det_size, min_overlap=0.7):
    """
    https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def preprocess_recognition(img, size=(96, 96)):
    # if len(img.shape) == 4:
    #     for i in range(img.shape[0]):
    #         if len(img[i, ::]) != 3:
    #             img_ = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #         img = cv2.resize(img, size)
    #
    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255
    return img


def get_images_of_faces():
    dir = 'data/infer_faces/'
    files = os.listdir(dir)
    imgs = []
    names = []
    for file in files:
        img = cv2.imread(dir + file)
        img = preprocess_recognition(img)
        imgs.append(img)
        names.append(file.split('.')[0])
    return np.stack(imgs), names