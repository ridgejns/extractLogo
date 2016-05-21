import imgPreprocessing
import cv2
import numpy as np
# from multiWindows import img2
    
if __name__ == '__main__':
    imgPATH = './img/'
    logoTp = cv2.imread(imgPATH+'purelogo256.png')
#     logoAffinePos = imgPreprocessing.LogoAffinePos(logoTp)
    logoAffinePos = imgPreprocessing.LogoAffinePos(logoTp,featureObject=cv2.AKAZE_create(), matcherObject=cv2.BFMatcher(), \
                                                    matchMethod = 'knnMatch')

    VWIDTH = 1280
    VHIGH = 720
#     VWIDTH = 960
#     VHIGH = 540
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,VWIDTH)
    ret = cap.set(4,VHIGH)
    ret,img = cap.read()
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()
    
    while(1):
        ret,imgRes = cap.read()
        img = imgRes.copy()
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgR = imgRes.copy()
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        logoContourPts, cPts, affinedcPts, affinedImg, rtnFlag = logoAffinePos.rcvAffinedAll(img)
        if (logoContourPts is not None):
            cv2.drawContours(img, [logoContourPts], -1, (0,255,0), 2)
        if (rtnFlag is True):
            affinedImgNarrow = cv2.resize(affinedImg,(0,0),fx=0.4,fy=0.4)
            cv2.imshow('affinedImg',affinedImgNarrow)
            cv2.imshow('croped',logoAffinePos.croped)
        
#         logoContourPts,logoContour , rtnFlag = logoAffinePos.extLegoLogo(img, minArea=0)
# #         cv2.imshow('mask',logoAffinePos.redmask)
# #         mask = cv2.cvtColor(logoAffinePos.redmask,cv2.COLOR_GRAY2BGR)
#         mask = cv2.resize(logoAffinePos.redmask,(0,0),fx=0.4,fy=0.4)
#         cv2.imshow('mask',mask)
# #         img3 = cv2.cvtColor(logoAffinePos.redmaskD,cv2.COLOR_GRAY2BGR)
# #         img3 = cv2.cvtColor(logoAffinePos.redmaskBlure,cv2.COLOR_GRAY2BGR)
#         if(rtnFlag is True):
#             cv2.drawContours(img, [logoContourPts], -1, (0,255,0), 2)
#             cPts, rtnFlag = logoAffinePos.extQuadrangleCpts(logoContourPts, logoContour)
# #             img4 = cv2.cvtColor(logoContour,cv2.COLOR_GRAY2BGR)
#             logoContour = cv2.resize(logoContour,(0,0),fx=0.4,fy=0.4)
#             cv2.imshow('logoContour',logoContour)
# #             print(rtnFlag)
#             if(rtnFlag is True):
#                 for idx, cPt in enumerate(cPts):
#                     cPt = cPt.flatten()
#                     img[cPt[1]-5:cPt[1]+5,cPt[0]-5:cPt[0]+5,:] = [255,255,0]
        
        
        imgNarrow = cv2.resize(img,(0,0),fx=0.4,fy=0.4)
        cv2.imshow('frame',imgNarrow)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cap.release()