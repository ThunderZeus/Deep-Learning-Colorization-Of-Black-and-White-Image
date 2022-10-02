import numpy as np 
import cv2
import matplotlib.pyplot as plt

prot_path = './model/colorization_deploy_v2.prototxt'
caffe_model = './model/colorization_release_v2.caffemodel'
point_path = './model/pts_in_hull.npy'

dnn_net = cv2.dnn.readNetFromCaffe(prot_path,caffe_model)
kernel = np.load(point_path)

Id = dnn_net.getLayerId("class8_ab")
Id2 = dnn_net.getLayerId("conv8_313_rh")

kernel = kernel.transpose().reshape(2,313,1,1)

dnn_net.getLayer(Id).blobs = [kernel.astype("float32")]
dnn_net.getLayer(Id2).blobs = [np.full([1,313],2.606,dtype='float32')]

image = cv2.imread('download.jpg')
image = image.astype("float32")/255.0

image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
