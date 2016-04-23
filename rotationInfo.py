import os
import cv2
import numpy as np

imgPATH = './img/'
dirMain = os.listdir(imgPATH)
dirSub = os.listdir(imgPATH+dirMain[1])

img = cv2.imread(imgPATH+dirMain[1]+'/'+dirSub[1])
# img = cv2.imread(imgPATH+'logo.png')
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgcanny = cv2.Canny(imggray,100,200)

logoimg = cv2.imread(imgPATH+'logo.png')
logoimg = cv2.resize(logoimg,(0,0),fx=0.2,fy=0.2)
logoimggray = cv2.cvtColor(logoimg,cv2.COLOR_BGR2GRAY)
logoimgHSV = cv2.cvtColor(logoimg,cv2.COLOR_BGR2HSV)
logoimgcanny = cv2.Canny(logoimggray,100,200)

imgW = imgHSV.shape[1]
imgH = imgHSV.shape[0]

lower_red=np.array([0,50,50])
upper_red=np.array([10,255,255])
mask1=cv2.inRange(imgHSV,lower_red,upper_red)
lower_red=np.array([170,50,50])
upper_red=np.array([179,255,255])
mask2=cv2.inRange(imgHSV,lower_red,upper_red)
mask = mask1+mask2

lower_red=np.array([0,50,50])
upper_red=np.array([10,255,255])
logomask1=cv2.inRange(logoimgHSV,lower_red,upper_red)
lower_red=np.array([170,50,50])
upper_red=np.array([179,255,255])
logomask2=cv2.inRange(logoimgHSV,lower_red,upper_red)
logomask = logomask1+logomask2

akaze = cv2.KAZE_create()

kp_akaze1, des_akaze1 = akaze.detectAndCompute(imggray, None)
kp_akaze2, des_akaze2 = akaze.detectAndCompute(logoimggray, None)

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
#  
# flann = cv2.FlannBasedMatcher(index_params,search_params)
#  
# matches = flann.knnMatch(des_akaze2,des_akaze1,k=2)
#  
# # matchesMask = [[0,0] for i in range(len(matches))]
#  
# good = []
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
# #         matchesMask[i]=[1,0]
#         good.append(m)
# 
# MIN_MATCH_COUNT = 10   
# if len(good)>MIN_MATCH_COUNT:
#     # 获取关键点的坐标
#     src_pts = np.float32([ kp_akaze2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp_akaze1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     # 第三个参数Method used to computed a homography matrix. The following methods are possible:
#     #0 - a regular method using all the points
#     #CV_RANSAC - RANSAC-based robust method
#     #CV_LMEDS - Least-Median robust method
#     # 第四个参数取值范围在1 到10，􁲁绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
#     # 超过误差就认为是outlier
#     # 返回值中M 为变换矩阵。
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# #     print(mask)
#     matchesMask = mask.ravel().tolist()
# #     print(matchesMask)
#     # 获得原图像的高和宽
#     h,w = logoimggray.shape
#     # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#     # 原图像为灰度图
#     cv2.polylines(logoimg,[np.int32(dst)],True,255,10, cv2.LINE_AA)
# else:
# #     print('Nononono')
#     print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#     matchesMask = None
#  
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# # img3 = cv2.drawMatchesKnn(img,kp_akaze1,logoimg,kp_akaze2,good,None,**draw_params)
# 
# # draw_params = dict(matchColor = (0,255,0),
# #                    singlePointColor = None,
# #                    matchesMask = matchesMask,
# #                    flags = 2)
# #  
# img3 = cv2.drawMatchesKnn(logoimg,kp_akaze2,img,kp_akaze1,good,None,**draw_params)


bf = cv2.BFMatcher()
matches = bf.match(des_akaze2,des_akaze1)
# matches = bf.knnMatch(des_akaze1,des_akaze2,k=2)
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# img3 = cv2.drawMatchesKnn(img,kp_akaze1,logoimg,kp_akaze1,good[:10],None,flags=2)
img3 = cv2.drawMatches(logoimg,kp_akaze2,img,kp_akaze1,matches[:10],None,flags=2)
# 
cv2.imshow('imgR',img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()