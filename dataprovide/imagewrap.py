import cv2
import numpy as np
import random
import sys
import time
from numba import cuda, jit


def random_crop(image, crop_size):
    h, w, d = img.shape
    x_offset = np.random.randint(low=0, high=h-crop_size, size=1)[0]
    y_offset = np.random.randint(low=0, high=w-crop_size, size=1)[0]
    cropped_img = image[x_offset:x_offset + crop_size, y_offset:y_offset + crop_size, ]
    return cropped_img


def normalize(data):
    mx = np.max(data)
    mn = np.min(data)
    return (data[:, :]-mn)/(mx - mn)


def wraping(img, mesh, crop_size, padding):
    min_x = sys.maxsize
    max_x = 0
    min_y = sys.maxsize
    max_y = 0
    result_size = crop_size + 2 * padding
    result = np.zeros([result_size, result_size, 3], dtype=np.int32)
    for i in range(result_size):
        for j in range(result_size):
            if (padding <= mesh[i, j, 1] <= crop_size+padding-1) & (padding <= mesh[i, j, 0] <= crop_size+padding-1):
                min_x = np.min([min_x, j])
                min_y = np.min([min_y, i])
                max_x = np.max([max_x, j])
                max_y = np.max([max_y, i])
                result[i, j, ] = img[int(mesh[i, j, 1]-padding), int(mesh[i, j, 0]-padding), ]
    return result[min_y:max_y, min_x:max_x, ]



def distored_mesh(size):
    col = np.arange(0, size)
    row = np.arange(0, size)
    col, row = np.meshgrid(row, col)
    mesh = np.stack((col, row), axis=-1)
    distance = np.abs(np.dot(mesh[:, :, ], [v_y, -v_x])-np.dot(point,  [v_y,-v_x]))
    distance = normalize(distance)
    distance = np.reshape(distance, [size, size, 1])
    #flod
    # alpha = [0.05, 0.1, 0.5]
    # weight = alpha/(mesh_d+alpha)

    # curl
    alpha = [1, 2, 5]
    weights = np.transpose(1-np.power(distance, alpha), [-1,0,1])

    weight = np.reshape(weights[2], [size*size, 1])
    weight = np.repeat(weight, 2, axis=-1)
    offset = np.reshape(weight*[v_x, v_y], [size, size, 2])
    mesh = mesh - offset
    return mesh


img = cv2.imread('../data/1.jpg')
crop_size = 1000
img = random_crop(img, crop_size)
point = np.random.randint(low=0, high=crop_size, size=2)[0:2]
v_x = random.uniform(-1, 1)
v_y = random.uniform(-1, 1)
v_x = v_x/(np.sqrt(v_x*v_x + v_y*v_y))
v_y = v_y/(np.sqrt(v_x*v_x + v_y*v_y))
v_x= v_x*200
v_y = v_y*200
padding = np.ceil(np.max(np.abs([v_y, v_x]))).astype(int)
result_size = crop_size + 2*padding
start = time.clock()
for i in range(100):
    mesh = distored_mesh(result_size)
elapsed = time.clock()-start
print(elapsed)
# start = time.clock()
# result = wraping(img, mesh, crop_size, padding)
# elapsed = time.clock()-start
# print(elapsed)
# cv2.imwrite('test.jpg', result)
