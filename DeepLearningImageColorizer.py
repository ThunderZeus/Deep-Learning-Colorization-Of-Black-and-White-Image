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

resized = cv2.resize(image,(224,224))

split = cv2.split(resized)[0]
split -= 50

dnn_net.setInput(cv2.dnn.blobFromImage(split))
forw = dnn_net.forward()[0, :, :, :].transpose((1,2,0))
 
forw = cv2.resize(forw, (image.shape[1],image.shape[0]))

split = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)
 
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

plt.imshow(colorized)
plt.axis('off');
