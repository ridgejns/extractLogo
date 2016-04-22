import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('pp.png')
# img = cv2.resize(img,(0,0),fx=0.1,fy=0.1)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
gray = np.float32(gray)
# # 输入图像必须是float32，最后一个参数在0.04 到0.05 之间
# dst = cv2.cornerHarris(gray,2,3,0.05)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
# # 返回的结果是[[ 311., 250.]] 两层括号的数组。
# corners = np.int0(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
# plt.imshow(img),plt.show()

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(img, contours, 3, (0,255,0), 3)
img = cv2.drawContours(img, contours, -1, (0,255,0), 1)
 
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()