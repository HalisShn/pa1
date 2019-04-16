# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:28:04 2019

@author: halis
"""

from helper import *
from Part1 import *
from scipy.cluster.vq import kmeans

def getSubpartsOfEdge(edge, k):
    edge_n = int(edge / k)
    return [(edge_n*i, edge_n*(i+1)) if edge_n*(i+1) < edge else (edge_n*i, edge) for i in range(k)]

def getSubImages(image, k):
    
    width, height = int(image.shape[1]), int(image.shape[0])

    width_parts, height_parts = getSubpartsOfEdge(width, k), getSubpartsOfEdge(height, k)

    sub_images = []
    for widths in width_parts:
        for heights in height_parts:
            img = image[heights[0]:heights[1],widths[0]:widths[1]]
            sub_images.append(img)
    return sub_images

def createCodebook(train_images, sift):
    
    train_desc = gen_sift_features(train_images, sift, get_desc = True)
    #codebook = train_desc[:500]
    if len(train_desc) > 500:
        #codebook, distortion = kmeans(train_desc, k_or_guess = 500, iter=1, thresh=1e-05)
        codebook = train_desc[:500]
        return codebook
    else:
        return train_desc

def getRepresentImages(codebook, images, sift, spatial_tiling = None):
    
    index_lists = []
    for img in images:
        #gray_img = to_gray(img)
        kp, desc = sift.detectAndCompute(img, None)
        if len(desc) > 500:
            #desc, distortion = kmeans(desc, k_or_guess = 500, iter=1, thresh=1e-05)
            desc = desc[0:500]
        index_list = getPredIndexes(codebook, desc)
        index_lists.append(index_list)
    return index_lists