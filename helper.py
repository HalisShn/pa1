# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:29:50 2019

@author: halis
"""
#==========================Libraries===============================#

from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from scipy.spatial import distance

#==================================================================#


#=======================File Functions=============================#

'''
Get Images' paths from given directories.
'''
def getImagesPaths(path):
    pathList = []
    for root, dirs, files in os.walk("content/drive/My\ Drive/uygulama/dataset/"+path):
        for file in files:
            path =  root +"\\" + file
            pathList.append(path)
    pathList.sort()        
    return pathList

'''
Get images from given paths.
'''
def getImages(img_paths):
    images = [cv2.imread(img_path) for img_path in img_paths]
    return images

'''
Get class number from given path string.
'''
def getClassNumberFromPath(nameList):
    classNumberList = []
    for name in nameList:   
        className = name.split("_")[0]
        classNumber = (int) (className[-3:])
        classNumberList.append(classNumber)
    return classNumberList

'''
Firstly: Get image paths from given directory name.
Secondly: Get images(array form) from given paths.
Thirdly: Get target class number from given path string.
'''
def getTargetandImagesFromPath(name):
    
    paths = getImagesPaths(name)
    images = getImages(paths)
    target = getClassNumberFromPath(paths)
    
    return images, target
 
#================================================================#


def getClassNameFromNumber(number):
    
    if number == 9:
        return "bear"
    elif number == 24:
        return "butterfly"
    elif number == 41:
        return "coffe-mug"
    elif number == 65:
        return "elk"
    elif number == 72:
        return "fire-truck"
    elif number == 105:
        return "horse"
    elif number == 107:
        return "hot-air-balloon"
    elif number == 118:
        return "iris"
    elif number == 152:
        return "owl"
    elif number == 212:
        return "teapot"
    
#==========================Accuracy=============================#

'''
This function prints class based accuracy and avarega accuracy of predicted 
target values that obtained from query images.
'''
def classBasedAccuracy(pred, target):
    list_classes = dict()
    for i in range(len(target)):
        class_name = getClassNameFromNumber(target[i])
        if i%5 == 0:
            list_classes[class_name] = 0
        if target[i] == pred[i]:
            list_classes[class_name] = list_classes[class_name] + 1
    print ("-------------------------------------------------------------")
    print ("| Class Name       |  Image Number  |  Score  | Accuracy(%) |")
    print ("-------------------------------------------------------------")
    
    
    for item in list_classes.items():
        accuracy = int(item[1]/5*100)
        if accuracy == 0:
            accuracy = " " + str(accuracy)
        else:
            accuracy = str(accuracy)
        print ("|", item[0], " "*(15-len(item[0])),
               "|"," "*5, 5," "*6,
               "|"," "*2, item[1]," "*2,
               "|"," "*3, accuracy," "*4,"|")
    print ("-------------------------------------------------------------")

    average = sum(list(list_classes.values()))
    accuracy = average * 2 
    print ("|", "Average", " "*(15-len("Average")),
               "|"," "*5, 50," "*5,
               "|"," "*2, average," "*1,
               "|"," "*3, accuracy," "*4,"|")
    print ("-------------------------------------------------------------")
    
#================================================================#

#============================Distance============================#

''' Calculate distance between two vector space '''
def getDistance(train_vector, query_vector, dist_func = "euclidean"):
    dist = 0
    if len(train_vector) != len(query_vector):
        a = len(train_vector)
        b = len(query_vector)
        if a < b:
            query_vector = query_vector[:a]
        else:
            train_vector = train_vector[:b]
    if dist_func == "euclidean":
        dist = distance.euclidean(train_vector, query_vector) 
    elif dist_func == "minkowski":
        dist = distance.minkowski(train_vector, query_vector)
    elif dist_func == "cosine":
        dist = distance.cosine(train_vector, query_vector)
    return dist

''' Get most similar images for each query images. '''
def getPredIndexes(train_vectors, query_vectors, dist_func = "euclidean"):
    index_list = []
    distances_list = []
    for i in range(len(query_vectors)):
        query_vctr = query_vectors[i]
        distances = []
        index = 0
        dist = getDistance(train_vectors[0], query_vctr, dist_func)
        #distances.append(dist)
        for j in range(1,len(train_vectors)):
            train_vctr = train_vectors[j]
            dist_new = getDistance(train_vctr, query_vctr, dist_func) 
            #distances.append(dist_new)
            if dist_new < dist:
                dist = dist_new
                index = j
        #distances_list.append(distances)
        index_list.append(index) 
    return index_list            

''' Get prediction class numbers from index list of train target. '''
def getPred(train_target, index_list):
    pred = []
    for index in index_list:
        pred.append(train_target[index])
    return pred

#================================================================#

def getMostSimilarImages(distances_list, train_images, methodName):
    
    print (len(distances_list))
    x = 0
    for distances in distances_list:
        distances2 = distances.copy()
        print (distances2[:5])
        distances2.sort()
        print (distances2[:5])
        indexes = [distances.index(i) for i in distances2[:5]]
        for i in range(len(indexes)):
            cv2.imwrite(methodName + str(x)+"-"+str(i)+".png", train_images[indexes[i]])
        x += 1
    
    
def printAccuracy(train_vector, train_target, query_vector, query_target, train_images, methodName, dist_func = "euclidean"):
    
    #index_list, distances_list = getPredIndexes(train_vector, query_vector, dist_func = dist_func)
    index_list = getPredIndexes(train_vector, query_vector, dist_func = dist_func)

    pred = getPred(train_target, index_list)
    
    print (query_target[0], query_target[5], query_target[10])
    print (pred[0], pred[5], pred[10])
    #getMostSimilarImages(distances_list[:11:5], train_images, methodName)
    

    classBasedAccuracy(pred, query_target)    
    
    
    
    
    
    
    
    