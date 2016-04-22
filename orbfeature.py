import scipy as sp
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

img1 = cv2.imread('logo.png')
img1 = cv2.resize(img1,(0,0),fx=0.2,fy=0.2)
# img1 = cv2.GaussianBlur(img1,(5,5),0)
img2 = cv2.imread('demo.png')
img2 = cv2.resize(img2,(0,0),fx=0.2,fy=0.2)
# img2 = cv2.GaussianBlur(img2,(5,5),0)
# # Initiate STAR detector
# orb = cv2.ORB_create()
# 
# # find the keypoints with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# 
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1,des2)
# 
# #matches = sorted(matches, key = lambda x:x.distance)
# #  draw only keypoints location,not size and orientation
# # img2 = cv2.drawKeypoints(sma,kp,sma,color=(0,255,0), flags=0)
img4 = copy.deepcopy(img1)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],img4,flags=2)

# orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
# orb = cv2.ORB_create()
# kp,des = orb.detectAndCompute(img,None)
# img2 = cv2.drawKeypoints(img,kp, None)
akaze = cv2.AKAZE_create()
# # 
# # kp_akaze = akaze.detect(img, None)
# # kp_akaze, des_akaze = akaze.compute(img, kp_akaze)
# # 
kp_akaze1, des_akaze1 = akaze.detectAndCompute(img1, None)
kp_akaze2, des_akaze2 = akaze.detectAndCompute(img2, None)
# print(des_akaze2.shape)
# print(kp_akaze1)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher()
matches = bf.match(des_akaze1,des_akaze2)
# matches = bf.knnMatch(des_akaze1,des_akaze2,k=2)
# print(matches)
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# iii=np.zeros([2,2])
        
iii=0
print(matches)
img3 = cv2.drawMatches(img1,kp_akaze1,img2,kp_akaze2,matches,iii,flags=2)
print(iii)

# cv2.drawKeypoints(matches)

# 
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50) # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des_akaze1,des_akaze2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0), 
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
# img3 = cv2.drawMatchesKnn(img1,kp_akaze1,img2,kp_akaze2,matches,None,**draw_params)


# # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# bf = cv2.BFMatcher()
# # matches = bf.knnMatch(des_akaze1,des_akaze2,k=2)
# img_akaze = cv2.drawKeypoints(img1,kp_akaze1,None)

# matches = sorted(matches, key = lambda x:x.distance)

# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# 
# # print(good)
# 
# img3 = cv2.drawMatchesKnn(img1,kp_akaze1,img2,kp_akaze2,good,flags=2)
# 
# 
cv2.imshow('dst',img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# plt.imshow(img2),plt.show()