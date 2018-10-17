# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 21:24:57 2018

@author: suxin
"""
import numpy as np
import gzip
from datetime import datetime


#C:\Users\suxin\.spyder\data\mnist
train_images_path='C:/Users/suxin/.spyder/data/mnist/train_images'
train_labels_path='C:/Users/suxin/.spyder/data/mnist/train_labels'
test_images_path='C:/Users/suxin/.spyder/data/mnist/test_images'
test_labels_path='C:/Users/suxin/.spyder/data/mnist/test_labels'


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
#仿照tensorflow中mnist.py写的
def extract_images(input_file, is_value_binary):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
#        if magic !=2051:
#            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))     
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print magic, num_images, rows, cols
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows*cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data
  

#抽取标签
#仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


train_images= extract_images(train_images_path,True)
train_images0= extract_images(train_images_path,False)
train_labels= extract_labels(train_labels_path)
test_images= extract_images(test_images_path,True)
test_images0= extract_images(test_images_path,False)
test_labels= extract_labels(test_labels_path)
#test_images,test_labels= load_mnist(test_images_path,test_labels_path)

def KNNClassify(new_input, train_images, train_labels, k):
    new_class=0
    m=train_images.shape[0]
    n=train_images.shape[1]
    new_input=new_input.reshape(1,n)
    dist=np.tile(new_input,(m,1))-train_images
    dist=np.sum(dist**2,axis=1)**0.5
    sorted_index=np.argsort(dist)
    class_count={}
    sorted_index=sorted_index[0:k]
    for num in sorted_index:
        if train_labels[num] in class_count:
            class_count[train_labels[num]]+=1
        else:
            class_count[train_labels[num]]=1
    new_class=max(class_count,key=class_count.get)            
    return new_class


from sklearn.decomposition import PCA
images=np.vstack((train_images0,test_images0))
pca=PCA(n_components=0.99)
pca.fit(images)
pca_images=pca.fit_transform(images)
pca_train_images=pca_images[0:train_images.shape[0]]
pca_test_images=pca_images[-test_images.shape[0]:]



#pca处理之后的数据
a = datetime.now()
numTestSamples = test_images.shape[0]
matchCount = 0
test_num = numTestSamples/10
for i in xrange(test_num):

    predict = KNNClassify(pca_test_images[i], pca_train_images, train_labels, 3)
    if predict == test_labels[i]:
        matchCount += 1
    if i % 100 == 0:
        print "完成%d张图片"%(i)
accuracy = float(matchCount) / test_num
b = datetime.now()
print"在原灰度值基础上使用PCA"
print "一共运行了%d秒"%((b-a).seconds)



print 'The classify accuracy is: %.2f%%' % (accuracy * 100)


#未经pca处理的数据
a = datetime.now()
numTestSamples = test_images.shape[0]
matchCount = 0
test_num = numTestSamples/10
for i in xrange(test_num):

    predict = KNNClassify(test_images[i], train_images, train_labels, 3)
    if predict == test_labels[i]:
        matchCount += 1
    if i % 100 == 0:
        print "完成%d张图片"%(i)
accuracy = float(matchCount) / test_num
b = datetime.now()
print"灰度值0-1二元处理"
print "一共运行了%d秒"%((b-a).seconds)


print 'The classify accuracy is: %.2f%%' % (accuracy * 100)
    
