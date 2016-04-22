import scipy as sp
import numpy as np
import cv2
from matplotlib import pyplot as plt

fileName = 'log1.png'
img = cv2.imread(fileName)
# img = cv2.resize(img,(0,0),fx=0.05,fy=0.05)
# print(img.shape)
img = cv2.resize(img,(254,144))
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
# img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
img = cv2.medianBlur(img,3)
# img = cv2.bilateralFilter(img,7,20,20)
# img = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret)
cannyedges = cv2.Canny(img,ret,200)
# sobelx64f = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=13)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
# ret,thresh = cv2.threshold(sobel_8u,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)




# gray = np.float32(gray)
# # 输入图像必须是float32，最后一个参数在0.04 到0.05 之间
# dst = cv2.cornerHarris(gray,2,3,0.05)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# cannyedges[dst>0.01*dst.max()]=[255]

# kernel = np.ones((1,1),np.uint8)
# erosion = cv2.erode(thresh,kernel,iterations = 1)
kernel = np.ones((9,9),np.uint8)
dilation = cv2.dilate(cannyedges,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 1)
# erosion = cv2.erode(dilation,kernel2,iterations = 1)
# dilation = cv2.dilate(erosion,kernel,iterations = 1)
# cannyedges2 = cv2.Canny(erosion,100,200)
# minLineLength = 10
# maxLineGap = 10
# lines = cv2.HoughLinesP(cannyedges,1,np.pi/180,60,minLineLength,maxLineGap)
# print(lines.shape)
# for i in range(lines.shape[0]):
#     for x1,y1,x2,y2 in lines[i]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# print(type(lines))

# lines = cv2.HoughLines(cannyedges2,1,np.pi/180,65)
# print(lines.shape)
# for i in range(lines.shape[0]):
#     for rho,theta in lines[i]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)

# gray = np.float32(dilation)
# # 输入图像必须是float32，最后一个参数在0.04 到0.05 之间
# dst = cv2.cornerHarris(gray,2,3,0.05)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# cannyedges[dst>0.01*dst.max()]=[255]

# image, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(img, contours, -1, (0,255,0), 1)

# cannyedges2 = cv2.Canny(erosion,100,200)

# fig=plt.figure()
# ax1=fig.subplot([1,2,1])
# ax1.imshow

cv2.imshow('dst',erosion)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# fig = plt.figure()
# a=fig.add_subplot(2,1,1)
# imgplot = plt.imshow(img)
# 
# b=fig.add_subplot(2,1,2)
# imgplot = plt.imshow(cannyedges2,cmap='gray')

plt.show()

# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

# cv2.imshow('img',cannyedges)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()