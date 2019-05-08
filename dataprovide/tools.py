import math
from numba import cuda
import numpy as np
import dataprovide.config as cfg
import cmath


@cuda.jit
def wrap_kernal(result, img, alpha, point_x, point_y, vector_x, vector_y, max, crop_size, padding, curl):
    i,j = cuda.grid(2)
    #计算定点到向量的距离
    distance = abs((j - point_x) * vector_y - (i - point_y) * vector_x)
    #每个位置的变形权值
    if curl == True:
        w = math.pow(distance/max, alpha)
    else:
        w = alpha/(distance/max + alpha)
    #变形网格
    mesh_x = int(j-w * vector_x * padding - padding)
    mesh_y = int(i-w * vector_y * padding - padding)
    if (0 <= mesh_y < crop_size) and (0 <= mesh_x < crop_size):
        result[i, j, 0] = img[mesh_y, mesh_x, 0]
        result[i, j, 1] = img[mesh_y, mesh_x, 1]
        result[i, j, 2] = img[mesh_y, mesh_x, 2]


def wrap_host(img, alpha, point_x, point_y, vector_x, vector_y, crop_size, padding, curl):

    size = crop_size + 2 * padding
    d_img = cuda.to_device(np.ascontiguousarray(img))
    d_result = cuda.device_array([size, size, 3], np.float32)

    threadsperblock = (cfg.TPB, cfg.TPB)
    blockspergrid_x = math.ceil(size / threadsperblock[0])
    blockspergrid_y = math.ceil(size / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    #4个定点的距离
    point_corner = np.array([[0, 0], [0, size-1], [size-1, 0], [size-1, size-1]], dtype= np.int32)
    distance_corner = abs(np.dot(point_corner-[point_x, point_y], [vector_y, -vector_x]))
    mx = max(distance_corner)


    wrap_kernal[blockspergrid, threadsperblock]\
        (d_result, d_img, alpha, point_x, point_y, vector_x, vector_y, mx, crop_size, padding, curl)

    return d_result.copy_to_host()