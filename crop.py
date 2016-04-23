import cv2
import numpy as np

img = cv2.imread('./img/purelogo.png')
img = cv2.resize(img,(1770,1770))

imgH,imgW,_ = img.shape
fH = int(imgH/3)
fW = int(imgW/3)
f1 = img[0:fH,0:fW]
f2 = img[fH+1:fH*2,0:fW]
f3 = img[fH*2+1:fH*3,0:fW]
f4 = img[0:fH,fW+1:fW*2]
f5 = img[fH+1:fH*2,fW+1:fW*2]
f6 = img[fH*2+1:fH*3,fW+1:fW*2]
f7 = img[0:fH,fW*2+1:fW*3]
f8 = img[fH+1:fH*2,fW*2+1:fW*3]
f9 = img[fH*2+1:fH*3,fW*2+1:fW*3]

cv2.imwrite('./img/f1.png',f1)
cv2.imwrite('./img/f2.png',f2)
cv2.imwrite('./img/f3.png',f3)
cv2.imwrite('./img/f4.png',f4)
cv2.imwrite('./img/f5.png',f5)
cv2.imwrite('./img/f6.png',f6)
cv2.imwrite('./img/f7.png',f7)
cv2.imwrite('./img/f8.png',f8)
cv2.imwrite('./img/f9.png',f9)