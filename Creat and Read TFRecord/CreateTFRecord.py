# -*- coding: UTF-8 -*- 

import matplotlib.pyplot as plt  
import os 
import cv2 
import numpy as np
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到

#判断文件是否为有效（完整）的图片
#输入参数为文件路径
#会出现漏检的情况
def IsValidImage(pathfile):
  bValid = True
  try:
    Image.open(pathfile).verify()
  except:
    bValid = False
  return bValid

cwd='/Users/yiqiwang/Desktop/TestDataSet/' 
classes={'husky','chihuahua'} #人为 设定 2 类
writer= tf.python_io.TFRecordWriter(cwd+"dog_train.tfrecords") #要生成的文件

for index,name in enumerate(classes):
    class_path=cwd+name+'/'
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name #每一个图片的地址
        value=IsValidImage(img_path)
        if not value:
            print (img_path)
            continue
        img=cv2.imread(img_path, 1)
        img=cv2.resize(img,(128,128))
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串

writer.close()