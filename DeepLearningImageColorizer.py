import numpy as np 
import cv2
import matplotlib.pyplot as plt

prot_path = './model/colorization_deploy_v2.prototxt'
caffe_model = './model/colorization_release_v2.caffemodel'
point_path = './model/pts_in_hull.npy'