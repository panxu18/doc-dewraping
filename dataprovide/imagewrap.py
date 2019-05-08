import cv2
import numpy as np
import random
import time
import math
from dataprovide.tools import wrap_host as wrap
import dataprovide.config as cfg


def random_crop(image, crop_size):
    h, w, d = image.shape
    x_offset = np.random.randint(low=0, high=h-crop_size, size=1)[0]
    y_offset = np.random.randint(low=0, high=w-crop_size, size=1)[0]
    return image[x_offset:x_offset + crop_size, y_offset:y_offset + crop_size, ]


def distored_mesh(img, crop_size, padding, alpha, point_x, point_y, vector_x, vector_y, curl):
    return wrap(img, alpha, point_x, point_y, vector_x, vector_y, crop_size, padding, True)


def image_warp():

    img = cv2.imread(cfg.image_path)
    img = random_crop(img, cfg.crop_size)

    point = np.random.randint(low=0, high=cfg.crop_size, size=2)[0:2]
    vector_x = random.uniform(-1, 1)
    vector_y = random.uniform(-1, 1)
    vector_mod = math.sqrt(vector_x*vector_x + vector_y*vector_y)
    vector_x = vector_x / vector_mod
    vector_y = vector_y / vector_mod

    # for i in range(15):
    #     cfg.alpha = 1.0 + i * 0.2
    #     result = wrap(img, cfg.alpha, point[0], point[1], vector_x, vector_y, cfg.crop_size, cfg.padding, cfg.iscurl)
    #     cv2.imwrite(str(round(cfg.alpha,1)) + 'test.jpg', result)

    paddings = np.array(cfg.paddings)
    for i in range(5):
        iscurl = np.random.randint(low=0, high=2, size=1)[0]
        N = len(paddings[iscurl])
        idx = np.random.randint(low=0, high=N, size=1)[0]
        alpha = paddings[iscurl][idx][0]
        padding = int(paddings[iscurl][idx][1])
        print(alpha, padding)
        result = wrap(img, alpha, point[0], point[1], vector_x, vector_y, cfg.crop_size, padding, iscurl == 1)
        cv2.imwrite('test' + str(iscurl) + str(round(alpha,1)) + str(padding) + '.jpg', result)


if __name__ == '__main__':
    # start = time.clock()

    image_warp()
    # elapsed = time.clock() - start
    # print('图像输出：', elapsed)
