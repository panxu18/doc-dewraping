# coding:utf-8
from wand.image import Image
import os

# 将pdf文件转为jpg图片文件
# ./PDF_FILE_NAME 为pdf文件路径和名称
filedir = '/home/xu-p/PycharmProjects/pdf2img/data'
savedir = '/home/xu-p/PycharmProjects/pdf2img/result'
list = os.listdir(filedir)
for i in range(0,len(list)):
    filename = os.path.join(filedir,list[i])
    if os.path.isfile(filename):
        _savedir = os.path.join(savedir,list[i])
        if os.path.exists(_savedir):
            continue
        os.mkdir(_savedir)
        image_pdf = Image(filename=filename, resolution=300)
        image_jpeg = image_pdf.convert('jpg')

        # wand已经将PDF中所有的独立页面都转成了独立的二进制图像对象。我们可以遍历这个大对象，并把它们加入到req_image序列中去。
        req_image = []
        for img in image_jpeg.sequence:
            img_page = Image(image=img)
            req_image.append(img_page.make_blob('jpg'))

        # 遍历req_image,保存为图片文件
        i = 0
        for img in req_image:
            saveFileName = str(i) + '.jpg'
            savepath = os.path.join(_savedir,saveFileName)
            ff = open(savepath, 'wb')
            ff.write(img)
            ff.close()
            i += 1