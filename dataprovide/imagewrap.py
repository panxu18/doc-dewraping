import cv2
import numpy as np
import random
# from matplotlib import pyplot as plt


def random_crop(image, crop_size):
    h, w, d = img.shape
    # 帮助随机裁剪和随机翻转一批图像
    cropped_img = np.zeros(h * w * d).reshape(h, w, d)


    x_offset = np.random.randint(low=0, high=h-crop_size, size=1)[0]
    y_offset = np.random.randint(low=0, high=w-crop_size, size=1)[0]
    cropped_img = image[x_offset:x_offset + crop_size,
                            y_offset:y_offset + crop_size, :]

    return cropped_img

def normalize(data):
    mx = np.max(data)
    mn = np.min(data)
    return (data[:,:]-mn)/(mx - mn)

img = cv2.imread('../data/1.jpg')
crop_size = 1000
img = random_crop(img, crop_size);
cv2.imwrite('croped.jpg', img)
point = np.random.randint(low=0, high=crop_size, size=2)[0:2]
print(point)
v_x = random.uniform(-1,1)
v_y = random.uniform(-1,1)
v_x = v_x/(np.sqrt(v_x*v_x + v_y*v_y))
v_y = v_y/(np.sqrt(v_x*v_x + v_y*v_y))
print(v_x,v_y)




# plt.matshow(weight[0], cmap='autumn')
# plt.matshow(weight[1],cmap='autumn')
# plt.matshow(weight[2],cmap='autumn')
# plt.show()
v_x= v_x*crop_size/1.4
v_y = v_y*crop_size/1.4
padding = np.ceil(np.max(np.abs([v_y, v_x]))).astype(int)
result_size = crop_size + 2*padding
result = np.zeros([result_size,result_size,3],dtype=np.int32)
mesh_x = np.arange(0,result_size)
mesh_y = np.arange(0,result_size)
mesh_x, mesh_y = np.meshgrid(mesh_y, mesh_x)
mesh = np.stack((mesh_x, mesh_y), axis=-1)
mesh_d = np.abs(np.dot(mesh[:,:,],[v_y,-v_x])-np.dot(point,[v_y,-v_x]))
mesh_d = normalize(mesh_d)
mesh_d = np.reshape(mesh_d,[result_size,result_size,1])
print(np.shape(mesh_d))
#flod
# alpha = [0.05,0.1,0.5]
# weight = alpha/(mesh_d+alpha)
# curl
alpha = [1,2,5]
weight = 1-np.power(mesh_d, alpha)

weight = np.transpose(weight, [-1,0,1])
mesh_xy = np.reshape(mesh, [result_size*result_size,2])
weight_xy = np.reshape(weight[2], [result_size*result_size,1])
weight_xy = np.repeat(weight_xy, 2, axis=-1)
offset = weight_xy *[v_x, v_y]
out_xy = mesh_xy - offset

out_xy = np.reshape(out_xy,[result_size, result_size,2])
for i in range(result_size):
    for j in range(result_size):
        if (padding <= out_xy[i,j,1] <= crop_size+padding-1 ) & (padding <= out_xy[i,j,0] <= crop_size+padding-1 ):
            result[i,j,] = img[int(out_xy[i,j,1]-padding),int(out_xy[i,j,0]-padding),]

cv2.imwrite('test.jpg', result)
