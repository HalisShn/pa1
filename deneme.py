# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:57:19 2019

@author: halis
"""
import numpy as np

list1 = [10,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

list2 = [2323,23223]

list2 += list1
print (list1)
print (list2)
print (list3)

list3 = np.arange(0, 300, 154)
print (list3, len(list3))


a = None 
if a:
    print ("adas")
    
a =3 
if a:
    print (a)
    
a, b = list3[0], list3[1]

print (a, b)