import imgPreprocessing
import cv2

if __name__ == '__main__':
    imgPATH = './img/'
    logoTp = cv2.imread(imgPATH+'purelogo256.png')
    logoAffinePos = imgPreprocessing.LogoAffinePos(logoTp)
#     logoAffinePos = imgPreprocessing.LogoAffinePos(logoTp,featureObject=cv2.AKAZE_create(), matcherObject=cv2.BFMatcher(), \
#                                                     matchMethod = 'knnMatch')
    VWIDTH = 960
    VHIGH = 540
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,VWIDTH)
    ret = cap.set(4,VHIGH)
    ret,img = cap.read()
    while(1):
        ret,imgRes = cap.read()
        img = imgRes.copy()
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgR = imgRes.copy()
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        logoContourPts, cPts, affinedcPts, affinedImg, rtnFlag = logoAffinePos.rcvAffinedAll(img)
#         print(rtnFlag)
        if (logoContourPts is not None):
            cv2.drawContours(img, [logoContourPts], -1, (0,255,0), 2)
        if (rtnFlag is True):
            img = affinedImg

        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cap.release()