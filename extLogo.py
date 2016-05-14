import cv2
import numpy as np

def extLegoLogo(img, minArea = 0):
    # It will extract the red area from the image
    imgH,imgW,_ = img.shape
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red=np.array([0,50,50])
    upper_red=np.array([10,255,255])
    mask1=cv2.inRange(imgHSV,lower_red,upper_red)
    lower_red=np.array([170,50,50])
    upper_red=np.array([179,255,255])
    mask2=cv2.inRange(imgHSV,lower_red,upper_red)
    redmask = mask1+mask2
    
    # It will try to get the logo contour
    _, contourPts, _ = cv2.findContours(redmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contoursPts = sorted(contourPts, key=cv2.contourArea, reverse=True)
    for idx, contourPts in enumerate(contoursPts):
        if idx > 5:
            # don't find any quadrangle area in first five contours, return false
            return None,None,None,False
        area = cv2.contourArea(contourPts)
        # estimated length of side
        estLength = np.sqrt(area)
        perimeter = cv2.arcLength(contourPts, True)
        if ((estLength * 4 <= perimeter * 1.15) & (estLength * 4 >= perimeter * 0.85)):
            # find a quadrangle area.
            # judge the area of this quadrangle, it must bigger than minArea, or return false
            if(area < minArea):
                return None,None,None,False
            logoContourPts = contourPts
            break
    logoContour = np.zeros([imgH,imgW],'uint8')
    cv2.drawContours(logoContour, [logoContourPts], -1, 255, 2)
    
    # corner points area
    cPtsAreaRtn = cv2.cornerHarris(logoContour.copy().astype('float32'),15,19,0.04)
    cPtsAreaRtn = cv2.dilate(cPtsAreaRtn, None)
    cPtsArea = np.zeros((imgH,imgW),'uint8')
    cPtsArea[cPtsAreaRtn > 0.01*cPtsAreaRtn.max()] = 255
    
    _, cptaContourPts, _ = cv2.findContours(cPtsArea, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # quadrangle can not find exactly 4 corners, return false
    if(len(cptaContourPts) != 4):
        return logoContourPts,logoContour,None,False
    cPts = np.zeros(8).reshape(4,2).astype('int32')
    for idx, cptaContourPt in enumerate(cptaContourPts):
        M = cv2.moments(cptaContourPt)
        p = np.int32([M['m10']/M['m00'],M['m01']/M['m00'] ])
        cPts[idx] = p
        
    cPts = cPts.reshape(-1,1,2)
    return logoContourPts,logoContour,cPts,True

def getRcvAffineInfo(logoContourPts, cPts, extraRotation = 0):
    M2 = cv2.moments(logoContourPts)
    logoCentrePt = np.int32([M2['m10']/M2['m00'],M2['m01']/M2['m00']])
    v = (cPts - logoCentrePt)
    pp = np.zeros(4).astype('uint8')
    ang00 = 0
    ang01 = __calcuVectorAngle__(v[pp[0]], v[1])
    ang02 = __calcuVectorAngle__(v[pp[0]], v[2])
    ang03 = __calcuVectorAngle__(v[pp[0]], v[3])
    vang = np.float32([ang00, ang01,ang02,ang03])
    pp[2] = abs(vang).argmax()
    vang[pp[2]] = 0
    pp[1] = vang.argmax()
    pp[3] = vang.argmin()
    print(extraRotation)
    if abs(extraRotation) < 10:
        pass
    elif (extraRotation > 80) & (extraRotation < 100) :
        pp = pp[[3,0,1,2]]
    elif (abs(extraRotation) > 170) & (abs(extraRotation) < 180) :
        pp = pp[[2,3,0,1]]
    elif (extraRotation > -100) & (extraRotation < -80) :
        pp = pp[[1,2,3,0]]
    else:
        print("invalied extraRotation")
        return None, None, True
        
    cPts = cPts[pp]
    area = cv2.contourArea(logoContourPts)
    # estimated half length of side
    estLength_half = np.sqrt(area) / 2
    destCPts_0 = np.int32([[[-estLength_half,estLength_half]],[[-estLength_half,-estLength_half]],\
                           [[estLength_half,-estLength_half]],[[estLength_half,estLength_half]]])
    destCPts = destCPts_0+logoCentrePt
    # recover matrix
    rcvM = cv2.getPerspectiveTransform(cPts.copy().astype('float32'), destCPts.copy().astype('float32'))
    return cPts, rcvM, True
    # change the cPts as general from [[[x0,y0]],[[x1,y1],[[x2,y2]],[[x3,y3]]]]
    
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
        # get matched key points
        src_pts = np.float64([ kpsrc[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts = np.float64([ kpdst[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        
        Ang = __calcuAngle__(src_pts,dst_pts)
        if(flag == 0):
            Ang = Ang/np.pi*180
    else:
        Ang = None
        print("Not enough matches are found - %d/%d" % (len(goodMatches),MIN_MATCH_COUNT))
    return Ang

def __calcuVectorAngle__(v1,v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    L1 = np.sqrt(v1.dot(v1))
    L2 = np.sqrt(v2.dot(v2))
    if((L1==0)|(L2==0)):
        Ang = None
    else:
        v1 = v1 / L1
        v2 = v2 / L2
        cosAng = v1.dot(v2)
        Ang = np.arccos(cosAng)
        if(np.cross(v1,v2)<0):
            Ang = -1*Ang
    return Ang

def __calcuAngle__(pts1,pts2,checks = None):
    checkstp = min(len(pts1),len(pts2))-1
    if checks is None:
        checks = checkstp
    else:
        checks = min(checks, checkstp)
#     Ang = np.zeros(checks)
    AngP = []
    AngN = []
    for i in range(checks):
        v1 = (pts1[i]-pts1[i+1])
        v2 = (pts2[i]-pts2[i+1])
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
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,VWIDTH)
    ret = cap.set(4,VHIGH)
    
    affinedcPtsSaved = np.zeros(8).reshape(-1,1,2)
    rcvMSave = 0
    cPtsSave = 0
#     --------frame write----------
#    Enable this part will start frame write.
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     out = cv2.VideoWriter()
#     succes = out.open('output.mp4v',fourcc, 10, (VWIDTH,VHIGH),True)
    ret,img = cap.read()
    while(1):
        ret,imgRes = cap.read()
        img = imgRes.copy()
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgR = imgRes.copy()
#         cropedImgLogo = np.zeros(10000).reshape(100,100)
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgH,imgW,_ = img.shape
        
        [logoContourPts, logoContour, cPts, rtnFlag] = extLegoLogo(img, minArea=7000)
        
        if(rtnFlag is True):
            cPrsRes = cPts.copy()
            cPts, rcvM, _ = getRcvAffineInfo(logoContourPts, cPts)
            img = imgRes.copy()
            imggray = cv2.warpPerspective(imggray, rcvM, (VWIDTH,VHIGH))
            affinedcPts = cv2.perspectiveTransform(cPts.copy().astype('float32'),rcvM)        
            xMax,yMax = affinedcPts.max(axis=0).flatten()
            xMin,yMin = affinedcPts.min(axis=0).flatten()
            cropedImgLogo = imggray[yMin-10:yMax+10,xMin-10:xMax+10]

            Ang = getRotInfo(cropedImgLogo, None, kaze, flann, kpsrc, dessrc)
            if Ang != None:
                cPts = cPrsRes.copy()
                cPts, rcvM, rtnFlag = getRcvAffineInfo(logoContourPts, cPts, -Ang)
                affinedcPts = cv2.perspectiveTransform(cPts.copy().astype('float32'),rcvM)
#                 print(affinedcPts)
#                 print(affinedcPts-affinedcPtsSaved)
                if(abs(affinedcPtsSaved - affinedcPts).sum() < 40):
                    affinedcPts = affinedcPtsSaved
                    rcvM = rcvMSave
                    cPts = cPtsSave
                 
                img = imgRes.copy()
                img = cv2.warpPerspective(img,rcvM, (VWIDTH,VHIGH))
                cv2.putText(img,str(Ang),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
                for i in range(4):    
                    cPt = affinedcPts[i].flatten()
                    img[cPt[1]-3:cPt[1]+3,cPt[0]-3:cPt[0]+3,:] = [255,0,255]
                    cv2.putText(img,str(i),(cPt[0],cPt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
#                 [logoContourPts,logoContour,cPts,rcvM,rtnFlag] = extLegoLogo(img,8000,-Ang)
#                 rtM = cv2.getRotationMatrix2D((imgW/2,imgH/2),Ang,1)
# # # #                 print(M)
#                 imgR = cv2.warpAffine(img,rcvM,(imgW,imgH))
#             print(Ang)
        else:
            cv2.putText(img,'No valid logo',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
#             print('No valid logo')
#             pass
#         out.write(img)
        
#         imgS = np.zeros([img.shape[0],img.shape[1]*2,img.shape[2]],'uint8')
#         imgS[:,0:img.shape[1],:] = imgRes
#         imgS[:,img.shape[1]:img.shape[1]*2,:] = img
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
    


