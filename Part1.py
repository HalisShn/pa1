# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:14:42 2019

@author: halis
"""

#==========================Libraries===============================#

import cv2
import numpy as np
from matplotlib import pyplot as plt

#==================================================================#

#=======================Gabor Filter Bank==========================#

def build_filters(ksize = 31, filter_number = 32):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / filter_number):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters):
    results = []
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        results.append(fimg)
    return results

def getMeansofFilteredImages(filtered_images):
    mean_list = []
    for img in filtered_images:
        mean = np.mean(cv2.mean(img)[:3])
        mean_list.append(mean)
    return mean_list

def getFeatureVectors(images, filters):
    vector_list = []
    for img in images:
        filtered_img = process(img, filters)
        mean_filtered_images = getMeansofFilteredImages(filtered_img)
        vector_list.append(mean_filtered_images)
    return vector_list

#==================================================================#

#==========================Average SIFT============================#

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(images, get_desc = False):

    sift = cv2.xfeatures2d.SIFT_create()
    print (np.shape(images))
    vector_list = []
    descriptors = []
    print (np.shape(images[0]))
    for img in images:
        #gray_img = to_gray(img)
        print ("image:", np.shape(img))
        kp, desc = sift.detectAndCompute(img, None)
        print (np.shape(desc))
        descriptors.append(desc)
        mean_desc = desc.mean(axis=0)
        print (np.shape(mean_desc))
        vector_list.append(mean_desc)
    if get_desc:
        descr = np.reshape(descriptors, (len(descriptors)//128, 128))
        descr = np.float32(descr)
        return descr
    else:
        
        return vector_list

#==================================================================#
 
