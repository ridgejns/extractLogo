import cv2
import numpy as np

img = cv2.imread('log2.jpg')
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

imgW = imgHSV.shape[1]
imgH = imgHSV.shape[0]

lower_red=np.array([0,50,50])
upper_red=np.array([15,255,255])
mask1=cv2.inRange(imgHSV,lower_red,upper_red)
lower_red=np.array([166,50,50])
upper_red=np.array([179,255,255])
mask2=cv2.inRange(imgHSV,lower_red,upper_red)

mask = mask1+mask2

# _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(img, contours, -1, (0,255,0), 2)
 
cv2.imshow('imgR',mask)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()