# import os
# import sys
import cv2
import numpy as np

def extLegoLogo(img):
#     if(img.ndim < 3):
#         sys.stderr.write('InputError: extLegoLogo(img) must pass a color image.\n')
# #         return None,None,None,False
#         sys.exit(1)
    
    imgH,imgW,_ = img.shape
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red=np.array([0,50,50])
    upper_red=np.array([10,255,255])
    mask1=cv2.inRange(imgHSV,lower_red,upper_red)
    lower_red=np.array([170,50,50])
    upper_red=np.array([179,255,255])
    mask2=cv2.inRange(imgHSV,lower_red,upper_red)
    mask = mask1+mask2
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print(len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    new_contours = []
    for idx, contour in enumerate(contours):
#         print(idx)
        if idx > 5:
            break
        # moment = cv2.moments(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if ((np.sqrt(area) * 4 <= perimeter * 1.15) & (np.sqrt(area) * 4 >= perimeter * 0.85)):
            new_contours.append(contour)
    if(len(new_contours) == 0):
        return None,None,None,False
#         break
#     print('box')
    logoContour = sorted(new_contours, key=cv2.contourArea, reverse=True)[0]
    # x,y,w,h = cv2.boundingRect(logocontour)
    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # print(cnt)
    # # compute the rotated bounding box of the contour
    rect = cv2.minAreaRect(logoContour)
    box = np.int0(cv2.boxPoints(rect))
    xaxis = np.array([box[0,0],box[1,0],box[2,0],box[3,0]])
    yaxis = np.array([box[0,1],box[1,1],box[2,1],box[3,1]])
    cropst = np.array([yaxis.min()-10,xaxis.min()-10])
    croped = np.array([yaxis.max()+10,xaxis.max()+10])
    crop = img[cropst[0]:croped[0],cropst[1]:croped[1]]
    mask = np.zeros([imgH,imgW]).astype('uint8')
    mask[cropst[0]:croped[0],cropst[1]:croped[1]] = 255
    H,W,_=crop.shape
    if((H<=100) | (W<=100)):
        return rect,mask,crop,False
    else:
        return rect,mask,crop,True

def getRotInfo(imggray,mask,kaze,flann,kpsrc,dessrc,flag = 0):

    if(imggray.ndim > 2):
        imggray = cv2.cvtColor(imggray,cv2.COLOR_BGR2GRAY)
    
    kpdst, desdst = kaze.detectAndCompute(imggray, mask)
    matches = flann.knnMatch(dessrc, desdst, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    goodMatches = []
#     matchesCounter = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
#             matchesCounter = matchesCounter+1
    #         if((matchesCounter == 1)|(matchesCounter == 2)|(matchesCounter == 3)):
    #             matchesMask1[i]=[1,0]
            goodMatches.append(m)

    MIN_MATCH_COUNT = 6
#     print(len(goodMatches))
    if len(goodMatches) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float64([ kpsrc[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts = np.float64([ kpdst[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        
        Ang = __calcuAngle__(src_pts,dst_pts)
        if(flag == 0):
            Ang = Ang/np.pi*180
        # 第三个参数Method used to computed a homography matrix. The following methods are possible:
        #0 - a regular method using all the points
        #CV_RANSAC - RANSAC-based robust method
        #CV_LMEDS - Least-Median robust method
        # 第四个参数取值范围在1 到10，􁲁绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
        # 超过误差就认为是outlier
        # 返回值中M 为变换矩阵。
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#         # 获得原图像的高和宽
#         h,w = logoimggray.shape
#         # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
#         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#         dst = cv2.perspectiveTransform(pts,M)
#         # 原图像为灰度图
#         cv2.polylines(img,[np.int32(dst)],True,[255,255,255],2, cv2.LINE_AA)
    else:
        Ang = None
        print("Not enough matches are found - %d/%d" % (len(goodMatches),MIN_MATCH_COUNT))
    return Ang

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
        v1 = (pts1[i]-pts1[i+1]).flatten()
        v2 = (pts2[i]-pts2[i+1]).flatten()
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
    Ang = sorted(Ang)
    Ang = Ang[rmNum:len(Ang)-rmNum]
    return np.average(Ang)
    
if __name__ == '__main__':
    imgPATH = './img/'
    logoimg = cv2.imread(imgPATH+'purelogo256.png')
#     logoimg = cv2.resize(logoimg,(0,0),fx=0.08,fy=0.08)
    logoimggray = cv2.cvtColor(logoimg,cv2.COLOR_BGR2GRAY)
    
    kaze = cv2.KAZE_create()
    kpsrc, dessrc = kaze.detectAndCompute(logoimggray, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
#     VWIDTH = 1280
#     VHIGH = 720
    VWIDTH = 960
    VHIGH = 540
#     VWIDTH = 640
#     VHIGH = 480
    cap=cv2.VideoCapture(0)
    ret = cap.set(3,VWIDTH)
    ret = cap.set(4,VHIGH)
    
#     --------frame write----------
#    Enable this part will start frame write.
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     out = cv2.VideoWriter()
#     succes = out.open('output.mp4v',fourcc, 10, (VWIDTH,VHIGH),True)
#     for i in range(1,6):
    while(1):
        ret,img = cap.read()
#         dirMain = os.listdir(imgPATH)
#         dm = dirMain[1]
#         dirSub = os.listdir(imgPATH+dm)
#         ds = dirSub[1]
#         img = cv2.imread(imgPATH+dm+'/'+ds)
#         # img = cv2.imread(imgPATH+'rot90.png')
#         img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
#         print(img.shape)
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgH,imgW,_ = img.shape
        
        extractedLogo = extLegoLogo(img)
#         if(extractedLogo[3] is True):
#             print('extrated logo is too close, please put further')
#             continue
        if(extractedLogo[3] is True):
            box = np.int0(cv2.boxPoints(extractedLogo[0]))
            cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
            cropgray = cv2.cvtColor(extractedLogo[2],cv2.COLOR_BGR2GRAY)
            Ang = getRotInfo(cropgray, None, kaze, flann, kpsrc, dessrc)
            if Ang != None:
                cv2.putText(img,str(Ang),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
#                 M = cv2.getRotationMatrix2D((imgW/2,imgH/2),Ang,1)
#                 print(M)
#                 img = cv2.warpAffine(img,M,(imgW,imgH))
#             print(Ang)
        else:
            cv2.putText(img,'No valid logo',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
#             print('No valid logo')
#             pass
#         out.write(img)


        
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
#     out.release()
    cv2.destroyAllWindows()


#     print(rect)
#     print(box)
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
#     M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
#     rot_img = cv2.warpAffine(img, M, (imgW,imgH))
#     # # print(imgH,imgW)
#     rectNew = ((rect[0][0],rect[0][1]),(rect[1][0]+4,rect[1][1]+4),0)
#     boxNew = np.int0(cv2.boxPoints(rectNew))
#     # print(rectNew)
#     # print(boxNew)
#     rot_img = cv2.drawContours(rot_img, [boxNew], -1, (0,255,0), 1)
#     cropst = np.array([rectNew[0][1]-rectNew[1][1]/2, rectNew[0][0]-rectNew[1][0]/2]).astype('uint64')
#     croped = np.array([rectNew[0][1]+rectNew[1][1]/2, rectNew[0][0]+rectNew[1][0]/2]).astype('uint64')
#     print(cropst,croped)
#     crop = rot_img[cropst[0]:croped[0],cropst[1]:croped[1]]
#     cropgray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
#     # cv2.imwrite('cropedlogo.png',crop)
#      
#     frag = cv2.imread('./img/cf10.png')
#     frag = cv2.resize(frag,(0,0),fx=0.2,fy=0.2)
#     fraggray = cv2.cvtColor(frag,cv2.COLOR_BGR2GRAY)
# 
#     cv2.imshow('imgR',crop)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()    
    


