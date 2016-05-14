"""
"""


import cv2
import numpy as np

class LogoAffinePos(object):
    def __init__(self, logoTemplate, featureObject=cv2.KAZE_create(), \
                 matcherObject=cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5), dict(checks=50)), \
                 matchMethod = 'knnMatch'):
        self.logoimg = logoTemplate
        self.logoimggray = cv2.cvtColor(self.logoimg,cv2.COLOR_BGR2GRAY)
        self.featureObj = featureObject
        self.kpsrc, self.dessrc = self.featureObj.detectAndCompute(self.logoimggray, None)
#         FLANN_INDEX_KDTREE = 0
#         index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#         search_params = dict(checks=50)   # or pass empty dictionary
        self.matcherObj = matcherObject
        self.matchMethod = matchMethod
        self.affinedcPtsSaved = np.zeros(8).reshape(-1,1,2)
        self.rcvMSave = 0
        self.cPtsSave = 0
  
    def extLegoLogo(self, img, minArea = 0):
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
        cv2.drawContours(logoContour, [logoContourPts], -1, 255, 3)
        
        # corner points area
        cPtsAreaRtn = cv2.cornerHarris(logoContour.copy().astype('float32'),int(estLength/5),15,0.04)
        cPtsAreaRtn = cv2.dilate(cPtsAreaRtn, None)
        cPtsArea = np.zeros((imgH,imgW),'uint8')
        cPtsArea[cPtsAreaRtn > 0.01*cPtsAreaRtn.max()] = 255
        
        # corner points area's contour points
        _, cptaContourPts, _ = cv2.findContours(cPtsArea, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # quadrangle can not find exactly 4 corners, return false
        if(len(cptaContourPts) is not 4):
            return logoContourPts,logoContour,None,False
        cPts = np.zeros(8).reshape(4,2).astype('int32')
        for idx, cptaContourPt in enumerate(cptaContourPts):
            M = cv2.moments(cptaContourPt)
            p = np.int32([M['m10']/M['m00'],M['m01']/M['m00'] ])
            cPts[idx] = p
            
        cPts = cPts.reshape(-1,1,2)
        return logoContourPts,logoContour,cPts,True
    
    
    def getRcvAffineInfo(self, logoContourPts, cPts, extraRotation = 0):
        M2 = cv2.moments(logoContourPts)
        logoCentrePt = np.int32([M2['m10']/M2['m00'],M2['m01']/M2['m00']])
        v = (cPts - logoCentrePt)
        pp = np.zeros(4).astype('uint8')
        ang00 = 0
        ang01 = self.__calcuVectorAngle__(v[pp[0]], v[1])
        ang02 = self.__calcuVectorAngle__(v[pp[0]], v[2])
        ang03 = self.__calcuVectorAngle__(v[pp[0]], v[3])
        vang = np.float32([ang00, ang01,ang02,ang03])
        pp[2] = abs(vang).argmax()
        vang[pp[2]] = 0
        pp[1] = vang.argmax()
        pp[3] = vang.argmin()
        
        angGap = 10
        if abs(extraRotation) < angGap:
            pass
        elif (extraRotation > (90-angGap)) & (extraRotation < (90+angGap)) :
            pp = pp[[3,0,1,2]]
        elif (abs(extraRotation) > (180-angGap)) & (abs(extraRotation) < 180) :
            pp = pp[[2,3,0,1]]
        elif (extraRotation > (-angGap-90)) & (extraRotation < (angGap-90)) :
            pp = pp[[1,2,3,0]]
        else:
            print("invalied extraRotation:",extraRotation)
            return None, None, False
        
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
    
    def getRotInfo(self,imggray,mask,flag = 0):
        if(imggray.ndim > 2):
            imggray = cv2.cvtColor(imggray,cv2.COLOR_BGR2GRAY)
        
        kpdst, desdst = self.featureObj.detectAndCompute(imggray, mask)
#         matches = self.matcherObj.knnMatch(self.dessrc, desdst, k=2)
        if (self.matchMethod is 'knnMatch'):
            matches = getattr(self.matcherObj,self.matchMethod)(self.dessrc, desdst, k=2)
            matchesMask = [[0,0] for i in range(len(matches))]
            goodMatches = []
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
                    goodMatches.append(m)
        elif (self.matchMethod is 'match'):
            goodMatches = getattr(self.matcherObj,self.matchMethod)(self.dessrc, desdst)
        else:
            print('invalid matchMethod setting')
            return None
#             print(len(goodMatches))
    
        MIN_MATCH_COUNT = 6
    #     print(len(goodMatches))
        if len(goodMatches) > MIN_MATCH_COUNT:
            # get matched key points
            src_pts = np.float64([ self.kpsrc[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            dst_pts = np.float64([ kpdst[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            Ang = self.__calcuAngle__(src_pts,dst_pts)
            if(flag == 0):
                Ang = Ang/np.pi*180
        else:
            Ang = None
#             print("Not enough matches are found - %d/%d" % (len(goodMatches),MIN_MATCH_COUNT))
        return Ang
    
    def __calcuVectorAngle__(self, v1,v2):
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
            cosAng = min(1,cosAng)
            cosAng = max(-1,cosAng)
            Ang = np.arccos(cosAng)
            if(np.cross(v1,v2)<0):
                Ang = -1*Ang
        return Ang

    def __calcuAngle__(self, pts1,pts2,checks = -1):
        checkstp = min(len(pts1),len(pts2))-1
        if checks < 0:
            checks = checkstp
        else:
            checks = min(checks, checkstp)
    #     Ang = np.zeros(checks)
        AngP = []
        AngN = []
        for i in range(checks):
            v1 = (pts1[i]-pts1[i+1])
            v2 = (pts2[i]-pts2[i+1])
            Ang = self.__calcuVectorAngle__(v1,v2)
            if(Ang is not None):
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
    
    def rcvAffinedAll(self,img):
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgH,imgW = imggray.shape
#         [logoContourPts, _, cPts, rtnFlag] = self.extLegoLogo(img)
        [logoContourPts, _, cPts, rtnFlag] = self.extLegoLogo(img, minArea=7000)
        if(rtnFlag is False):
            return logoContourPts, cPts, None, None, False
        else:
            cPrsRes = cPts.copy()
            cPts, rcvM, _ = self.getRcvAffineInfo(logoContourPts, cPts)
            imggray = cv2.warpPerspective(imggray, rcvM, (imgW,imgH))
            affinedcPts = cv2.perspectiveTransform(cPts.copy().astype('float32'),rcvM)
            xMax,yMax = affinedcPts.max(axis=0).flatten() + 10
            xMin,yMin = affinedcPts.min(axis=0).flatten() - 10
            if((xMin<0) | (yMin<0) | (xMax>imgW) | (yMax>imgH)):
                return logoContourPts, cPts, affinedcPts, rcvM, False
            
            cropedImgLogo = imggray[yMin:yMax,xMin:xMax]
            
            Ang = self.getRotInfo(cropedImgLogo, None)
#             print(Ang)
            if Ang is None:
                return logoContourPts, cPts, affinedcPts, rcvM, False
            else:
                cPts = cPrsRes.copy()
                cPts, rcvM, rtnFlag = self.getRcvAffineInfo(logoContourPts, cPts, -Ang)
                if (rtnFlag is False):
                    return logoContourPts, cPts, affinedcPts, rcvM, False
                affinedcPts = cv2.perspectiveTransform(cPts.copy().astype('float32'),rcvM)
                return logoContourPts, cPts, affinedcPts, rcvM, True


        