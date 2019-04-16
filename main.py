# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:54:33 2019

@author: halis
"""

#==========================Libraries===============================#

from helper import *
from Part1 import *
from Part2 import *
from scipy.cluster.vq import vq, kmeans, whiten

#==================================================================#

''' Get images and traget value of images '''
train_images, train_target = getTargetandImagesFromPath("train")
print ("Train images uploaded.")
query_images, query_target = getTargetandImagesFromPath("query")
print ("Query images uploaded." )


'''
#=======================Gabor Filter Bank==========================#

print ("Gabor Filter Bank")
filters = build_filters()

train_vectors = getFeatureVectors(train_images, filters)
query_vectors = getFeatureVectors(query_images, filters)

printAccuracy(train_vectors, train_target, query_vectors, query_target, train_images, "gabor", dist_func = "minkowski")
    

#==================================================================#

#==================================================================#
print ("Average SIFT")
'''

sift = cv2.xfeatures2d.SIFT_create()

train_vectors = gen_sift_features(train_images, sift)
#query_vectors = gen_sift_features(query_images, sift)

#printAccuracy(train_vectors, train_target, query_vectors, query_target, train_images, "sift", dist_func = "minkowski")

#==================================================================#


# PART 2
#===========================Code Book==============================#
'''
codebook = createCodebook(train_images, sift)
print ("Codebook created")

reps_train_images = getRepresentImages(codebook, train_images, sift)
print ("reps_train")

reps_query_images = getRepresentImages(codebook, query_images, sift)
print ("reps_query")


printAccuracy(reps_train_images, train_target, reps_query_images, query_target, train_images, "bow")
'''

