# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:11:00 2016

@author: abulin
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

img = cv2.imread('log1.jpg',1)
#number = cv2.imread('lego.jpg',0)
#number = cv2.resize(number, (0,0), fx=1.5, fy=1.5)
small = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
sma= copy.deepcopy(small)
height = np.size(small, 0)
width = np.size(small, 1)
threshold=70
sma = cv2.GaussianBlur(sma,(3,3),0)
for e in range(0,3):
    for i in range(1,height-1):
        for l in range(1,width-1):
            if sma[i,l,e]<=threshold:
                sma[i,l,e]=0
            else: sma[i,l,e]=255


for k in range(1,height-1):
    for m in range(1,width-1):
        if sma[k,m,0]+sma[k,m,1]+sma[k,m,2]!=0:
            sma[k,m,0]=255
            sma[k,m,1]=255
            sma[k,m,2]=255

for a in range(1,height-1):
    for b in range(1,width-1):
        if sma[a,b,0]==0:
            if sma[a-1,b-1,0]!=0:
                sma[a-1,b-1,1]=0
            if sma[a-1,b,0]!=0:
                sma[a-1,b,1]=0
            if sma[a-1,b+1,0]!=0:
                sma[a-1,b+1,1]=0            
            if sma[a,b-1,0]!=0:
                sma[a,b-1,1]=0                
            if sma[a,b+1,0]!=0:
                sma[a,b+1,1]=0                                
            if sma[a+1,b-1,0]!=0:
                sma[a+1,b-1,1]=0
            if sma[a+1,b,0]!=0:
                sma[a+1,b,1]=0
            if sma[a+1,b+1,0]!=0:
                sma[a+1,b+1,1]=0                
#sma = cv2.Canny(sma,100,200)
#number = cv2.Canny(number,100,200)
    
# Initiate STAR detector
#orb = cv2.ORB_create()

# find the keypoints with ORB
#kp1, des1 = orb.detectAndCompute(small,None)
#kp2, des2 = orb.detectAndCompute(number,None)

#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#matches = bf.match(des1,des2)

#matches = sorted(matches, key = lambda x:x.distance)
# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(sma,kp,sma,color=(0,255,0), flags=0)
#img4 = copy.deepcopy(sma)
#img3 = cv2.drawMatches(sma,kp1,number,kp2,matches[:10],img4,flags=2)
#cv2.imwrite('legosma.jpg',sma)

#cv2.imshow('img1',img2)
#cv2.waitKey()
#cv2.imshow('img2',sma)
#cv2.waitKey()
#cv2.destroyAllWindows()
cv2.imshow('img3',sma) 
cv2.waitKey()
cv2.destroyAllWindows() 