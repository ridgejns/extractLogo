import os
import cv2
import numpy as np

imgPATH = './img/'
dirMain = os.listdir(imgPATH)
dm = dirMain[1]
dirSub = os.listdir(imgPATH+dm)
ds = dirSub[2]
img = cv2.imread(imgPATH+dm+'/'+ds)
# img = cv2.imread(imgPATH+'logo3.png')
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
imgH,imgW,_ = img.shape

def extLegoLogo(img):
#     imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     imgcanny = cv2.Canny(imggray,100,200)
#     
#     imgW = imgHSV.shape[1]
#     imgH = imgHSV.shape[0]
    
    lower_red=np.array([0,50,50])
    upper_red=np.array([10,255,255])
    mask1=cv2.inRange(imgHSV,lower_red,upper_red)
    lower_red=np.array([170,50,50])
    upper_red=np.array([179,255,255])
    mask2=cv2.inRange(imgHSV,lower_red,upper_red)
    mask = mask1+mask2
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    new_contours = []
    for idx, contour in enumerate(contours):
        if idx > 5:
            break
        # moment = cv2.moments(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if ((np.sqrt(area) * 4 <= perimeter * 1.1) & (np.sqrt(area) * 4 >= perimeter * 0.9)):
            new_contours.append(contour)
    
    logoContour = sorted(new_contours, key=cv2.contourArea, reverse=True)[0]
    
    # x,y,w,h = cv2.boundingRect(logocontour)
    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # print(cnt)
    # # compute the rotated bounding box of the contour
    rect = cv2.minAreaRect(logoContour)
    return rect

rect = extLegoLogo(img)
box = np.int0(cv2.boxPoints(rect))
print(rect)
print(box)

# tp1 = np.array([box[0,0]/3*2+box[1,0]/3,box[0,1]/3*2+box[1,1]/3])
# tp2 = np.array([box[0,0]/3+box[1,0]/3*2,box[0,1]/3+box[1,1]/3*2])
# tp3 = np.array([box[3,0]/3+box[2,0]/3*2,box[3,1]/3+box[2,1]/3*2])
# tp4 = np.array([box[3,0]/3*2+box[2,0]/3,box[3,1]/3*2+box[2,1]/3])
# assumeAr1 = np.array([tp1,tp2,tp3,tp4]).astype('uint64')
# 
# tp1 = np.array([box[1,0]/3*2+box[2,0]/3,box[1,1]/3*2+box[2,1]/3])
# tp2 = np.array([box[1,0]/3+box[2,0]/3*2,box[1,1]/3+box[2,1]/3*2])
# tp3 = np.array([box[0,0]/3+box[3,0]/3*2,box[0,1]/3+box[3,1]/3*2])
# tp4 = np.array([box[0,0]/3*2+box[3,0]/3,box[0,1]/3*2+box[3,1]/3])
# 
# assumeAr2 = np.array([tp1,tp2,tp3,tp4]).astype('uint64')
# 
# # print(assumeAr2)
# img = cv2.drawContours(img, [box], -1, (0,255,0), 1)
# # img = cv2.drawContours(img, [assumeAr1], -1, (0,255,0), 1)
# img = cv2.drawContours(img, [assumeAr2], -1, (0,255,0), 1)

# rotate_angle = rect[2]
M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
rot_img = cv2.warpAffine(img, M, (imgW,imgH))
# print(imgH,imgW)
rectNew = ((rect[0][0],rect[0][1]),(rect[1][0]+4,rect[1][1]+4),0)
boxNew = np.int0(cv2.boxPoints(rectNew))
print(rectNew)
print(boxNew)
# rot_img = cv2.drawContours(rot_img, [boxNew], -1, (0,255,0), 1)
cropst = np.array([rectNew[0][1]-rectNew[1][1]/2, rectNew[0][0]-rectNew[1][0]/2]).astype('uint64')
croped = np.array([rectNew[0][1]+rectNew[1][1]/2, rectNew[0][0]+rectNew[1][0]/2]).astype('uint64')
print(cropst,croped)
crop = rot_img[cropst[0]:croped[0],cropst[1]:croped[1]]

kaze = cv2.KAZE_create()


frag = cv2.imread('./img/f8.png')
frag = cv2.resize(frag,(0,0),fx=0.08,fy=0.08)
w, h, _ = frag.shape
method = cv2.TM_CCOEFF
res = cv2.matchTemplate(crop,frag,cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc

bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(crop,top_left, bottom_right, 255, 1)

# kp1, des1 = kaze.detectAndCompute(frag, None)
# kp2, des2 = kaze.detectAndCompute(crop, None)
# 
# bf = cv2.BFMatcher()
# matches = bf.match(des1,des2)
# img3 = cv2.drawMatches(frag,kp1,crop,kp2,matches[:10],None,flags=0)

cv2.imshow('imgR',crop)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()