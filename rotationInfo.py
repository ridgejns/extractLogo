import os
import cv2
import numpy as np
from extLogo import extLegoLogo

def __calcuVectorAngle__(v1,v2):
    L1 = np.sqrt(v1.dot(v1))
    L2 = np.sqrt(v2.dot(v2))
    if((L1==0)|(L2==0)):
        Ang = None
    else:
        v1 = v1/ L1
        v2 = v2/ L2
        cosAng = v1.dot(v2)
        Ang = np.arccos(cosAng)
        if(np.cross(v1,v2)<0):
            Ang = -1*Ang
    return Ang

def __calcuAngle__(pts1,pts2,checks=None):
    checkstp = min(len(pts1),len(pts2))-1
    if checks is None:
        checks = checkstp
    else:
        checks = min(checks, checkstp)
#     Ang = np.zeros(checks)
    AngP = []
    AngN = []
    for i in range(checks):
#         print(i)
        v1 = (pts1[i]-pts1[i+1]).flatten()
        v2 = (pts2[i]-pts2[i+1]).flatten()
#         print(v1,v2)
        Ang = __calcuVectorAngle__(v1,v2)
        if(Ang!=None):
            if(Ang<0):
                AngN.append(Ang)
            else:
                AngP.append(Ang)
    if(len(AngP)>len(AngN)):
        Ang = AngP
    else:
        Ang = AngN
    rmNum = int(np.ceil(len(Ang)*0.2))
    sorted(Ang)
    Ang = Ang[rmNum:len(Ang)-rmNum]
    return np.average(Ang)

imgPATH = './img/'
dirMain = os.listdir(imgPATH)
dm = dirMain[5]
dirSub = os.listdir(imgPATH+dm)
ds = dirSub[1]
img = cv2.imread(imgPATH+dm+'/'+ds)
# img = cv2.imread(imgPATH+'rotm10.png')
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# imgcanny = cv2.Canny(imggray,100,200)

logoimg = cv2.imread(imgPATH+'purelogo128.png')
# logoimg = cv2.resize(logoimg,(0,0),fx=0.1,fy=0.1)
logoimggray = cv2.cvtColor(logoimg,cv2.COLOR_BGR2GRAY)
# logoimgHSV = cv2.cvtColor(logoimg,cv2.COLOR_BGR2HSV)
# logoimgcanny = cv2.Canny(logoimggray,100,200)

extractedLogo = extLegoLogo(img)
# box = np.int0(cv2.boxPoints(rect))
# xaxis = np.array([box[0,0],box[1,0],box[2,0],box[3,0]])
# yaxis = np.array([box[0,1],box[1,1],box[2,1],box[3,1]])
# cropst = np.array([yaxis.min()-10,xaxis.min()-10])
# croped = np.array([yaxis.max()+10,xaxis.max()+10])
# crop = img[cropst[0]:croped[0],cropst[1]:croped[1]]
cropgray = cv2.cvtColor(extractedLogo[2],cv2.COLOR_BGR2GRAY)
# mask = np.zeros([imgH,imgW]).astype('uint8')
# mask[cropst[0]:croped[0],cropst[1]:croped[1]] = 255

kaze = cv2.KAZE_create()

kp1, des1 = kaze.detectAndCompute(logoimggray, None)
# kp2, des2 = kaze.detectAndCompute(cropgray, None)
kp2, des2 = kaze.detectAndCompute(imggray, extractedLogo[1])

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0,0] for i in range(len(matches))]
# matchesMask1 = [[0,0] for i in range(len(matches))]
# matchesMask2 = [[0,0] for i in range(len(matches))]
#  
goodMatches = []
matchesCounter = 0
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        matchesCounter = matchesCounter+1
#         if((matchesCounter == 1)|(matchesCounter == 2)|(matchesCounter == 3)):
#             matchesMask1[i]=[1,0]
        goodMatches.append(m)
# 
MIN_MATCH_COUNT = 5
print(len(goodMatches))
if len(goodMatches) > MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float64([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
    dst_pts = np.float64([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
    # 第三个参数Method used to computed a homography matrix. The following methods are possible:
    #0 - a regular method using all the points
    #CV_RANSAC - RANSAC-based robust method
    #CV_LMEDS - Least-Median robust method
    # 第四个参数取值范围在1 到10，􁲁绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
    # 超过误差就认为是outlier
    # 返回值中M 为变换矩阵。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     print(mask)
#     matchesMask = mask.ravel().tolist()
#     print(matchesMask)
    # 获得原图像的高和宽
    h,w = logoimggray.shape
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    # 原图像为灰度图
    cv2.polylines(img,[np.int32(dst)],True,[255,255,255],2, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(goodMatches),MIN_MATCH_COUNT))

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                    matchesMask = matchesMask,
#                     matchesMask = None,
                   flags = 2)

Ang = __calcuAngle__(src_pts,dst_pts)
Ang = Ang/np.pi*180
print(Ang)
font=cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,str(Ang),(10,500),font, 4,(255,255,255),2)
img3 = cv2.drawMatchesKnn(logoimg,kp1,img,kp2,matches,None,**draw_params)

cv2.imshow('imgR',img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()