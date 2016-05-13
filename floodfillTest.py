import cv2
import numpy as np

if __name__ == '__main__':
    blankimg = cv2.imread("./img/blankimg.png",1)
    gray = cv2.cvtColor(blankimg,cv2.COLOR_BGR2GRAY)
    bk = np.zeros(gray.shape,'uint8')
    mask = np.zeros(gray.shape+np.array([2,2]),'uint8')
#     diff = (6,6,6)
#     cv2.floodFill(gray,mask,(400,340),255)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,20,7,0.04)
    dst = cv2.dilate(dst,None)
    bk[dst>0.01*dst.max()]=255
    
    _, contours, _ = cv2.findContours(bk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blankimg,contours,-1,(0,255,0), 1)
    corner = []
    for idx, contour in enumerate(contours):  
        cnt = contour
        M = cv2.moments(cnt)
        p = [int(M['m10']/M['m00']),int(M['m01']/M['m00'])]
        corner.append(p)
        blankimg[p[1]-3:p[1]+3,p[0]-3:p[0]+3,:] = [0,0,255]
        
#     cv2.imshow('blankimg',blankimg)
#     if cv2.waitKey(0) & 0xFF == 27:
#         cv2.destroyAllWindows()