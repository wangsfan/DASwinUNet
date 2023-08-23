import glob
import cv2
import numpy as np
import os
from skimage.io import imread

def npz(im, la, s):
    images_path = im
    labels_path = la
    path2 = s
    images = os.listdir(images_path)
    for s in images:
        print(s)
        image_path = os.path.join(images_path, s)
        label_path = os.path.join(labels_path, s)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		# 标签由三通道转换为单通道
        label = cv2.imread(label_path, flags=0)
        label = cv2.resize(label, (224, 224))
        label = np.array(label)
        label = label/255
        #print(label)
        # 保存npz文件
        np.savez(path2+s[:-4]+".npz",image=image,label=label)

npz('Image Path', 'Label Path', 'npz file save path')

