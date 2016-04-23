import os
import cv2
import numpy as np

imgPATH = './img/'
dirMain = os.listdir(imgPATH)
dirSub = os.listdir(imgPATH+dirMain[1])

img = cv2.imread(imgPATH+dirMain[1]+'/'+dirSub[3])
# img = cv2.imread(imgPATH+'logo.png')
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgcanny = cv2.Canny(imggray,100,200)

# 
# logoimg = cv2.imread(imgPATH+'logo2.png')
# logoimg = cv2.resize(logoimg,(0,0),fx=0.2,fy=0.2)
# logoimggray = cv2.cvtColor(logoimg,cv2.COLOR_BGR2GRAY)
# logoimgHSV = cv2.cvtColor(logoimg,cv2.COLOR_BGR2HSV)
# logoimgcanny = cv2.Canny(logoimggray,100,200)

imgW = imgHSV.shape[1]
imgH = imgHSV.shape[0]

lower_red=np.array([0,50,50])
upper_red=np.array([10,255,255])
mask1=cv2.inRange(imgHSV,lower_red,upper_red)
lower_red=np.array([170,50,50])
upper_red=np.array([179,255,255])
mask2=cv2.inRange(imgHSV,lower_red,upper_red)
mask = mask1+mask2

# lower_red=np.array([0,50,50])
# upper_red=np.array([10,255,255])
# logomask1=cv2.inRange(logoimgHSV,lower_red,upper_red)
# lower_red=np.array([170,50,50])
# upper_red=np.array([179,255,255])
# logomask2=cv2.inRange(logoimgHSV,lower_red,upper_red)
# logomask = logomask1+logomask2



 
cv2.imshow('imgR',mask)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()