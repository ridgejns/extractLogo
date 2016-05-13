import cv2
import numpy as np
import extLogo

if __name__ == '__main__':
    imgPATH = './img/'
    logoimg = cv2.imread(imgPATH+'purelogo128.png')
    logoimggray = cv2.cvtColor(logoimg,cv2.COLOR_BGR2GRAY)
    ym, xm = logoimggray.shape
    cornerT = np.float32([[0,ym],[0,0],[xm,0],[xm,ym]])
    cornerSave = np.zeros(8).reshape(4,2)
#     cornerT = np.float32([[0,ym],[0,0],[xm,0]])
    kaze = cv2.KAZE_create()
    kpsrc, dessrc = kaze.detectAndCompute(logoimggray, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
#     VWIDTH = 1280
#     VHIGH = 720
#     VWIDTH = 960
#     VHIGH = 540
    VWIDTH = 640
    VHIGH = 480
    cap=cv2.VideoCapture(0)
    ret = cap.set(3,VWIDTH)
    ret = cap.set(4,VHIGH)
    maskNewNew = np.zeros([VHIGH,VWIDTH],np.uint8)
    while(1):
        ctline = np.zeros([VHIGH,VWIDTH])
        ret,frame = cap.read()
        framegray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame2 = np.zeros(framegray.shape)
        frameH,frameW, = framegray.shape
        
        imgHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_red=np.array([0,50,50])
        upper_red=np.array([10,255,255])
        mask1=cv2.inRange(imgHSV,lower_red,upper_red)
        lower_red=np.array([170,50,50])
        upper_red=np.array([179,255,255])
        mask2=cv2.inRange(imgHSV,lower_red,upper_red)
        mask = mask1+mask2
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         print('CT')
#         print(contours)
#         print('CT')
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
        
        if(len(new_contours)>0):
            logoContour = sorted(new_contours, key=cv2.contourArea, reverse=True)[0]
#             logoContour = sorted(new_contours, key=cv2.contourArea, reverse=True)
#             cv2.drawContours(frame, new_contours, -1, (0,255,0), 2)
#             print('\nCT\n')
#             print(logoContour)
            cv2.drawContours(frame, [logoContour], -1, (255,255), 2)
            cv2.drawContours(ctline,[logoContour], -1, (255,255), 2)
#             mask = np.zeros(ctline.shape+np.array([2,2]),'uint8')
#             cv2.floodFill(ctline,mask,(400,340),255)
            ctline = np.float32(ctline)
            
            # the parameter probably need to change
            dst = cv2.cornerHarris(ctline,15,19,0.04)
            # the parameter probably need to change
            dst = cv2.dilate(dst,None)
            bk = np.zeros(ctline.shape,'uint8')
            bk[dst>0.01*dst.max()]=255
            _, contours2, _ = cv2.findContours(bk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours2) == 4:
                cv2.drawContours(frame,contours2,-1,(0,255,0), 1)
                corner = []
                for idx, contour in enumerate(contours2):  
                    cnt = contour
                    M = cv2.moments(cnt)
                    p = [int(M['m10']/M['m00']),int(M['m01']/M['m00'])]
                    corner.append(p)
#                     frame[p[1]-3:p[1]+3,p[0]-3:p[0]+3,:] = [255,0,255]
                corner = np.asarray(corner,'int32')
                if(np.average(abs((corner-cornerSave))) < 5):
                    corner = cornerSave
                cornerSave = corner
                M = cv2.moments(logoContour)
                cent = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
#                 print(cent)
                v = corner - cent
                p0 = 0
                ang1=extLogo.__calcuVectorAngle__(v[p0], v[1])
                ang2=extLogo.__calcuVectorAngle__(v[p0], v[2])
                ang3=extLogo.__calcuVectorAngle__(v[p0], v[3])
                vang = np.array([ang1,ang2,ang3])
#                 print(vang)
                p2 = abs(vang).argmax()
                vang[p2] = 0
                p2 = p2+1
                p1 = vang.argmax()+1
                p3 = vang.argmin()+1
                cornerI = corner[[p0,p1,p2,p3]].astype('float32')
                print(cornerI)
                print(cent)
                print("----------------")
#                 cornerI = np.float32([corner[p0],corner[p1],corner[p2],corner[p3]])
#                 cornerI = np.float32([corner[p0],corner[p1],corner[p2]])
                cv2.putText(frame,'1',(cornerI[0][0],cornerI[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
                cv2.putText(frame,'2',(cornerI[1][0],cornerI[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
                cv2.putText(frame,'3',(cornerI[2][0],cornerI[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
                cv2.putText(frame,'4',(cornerI[3][0],cornerI[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
                cornerI = cornerI-cornerI.min(axis=0)
                M = cv2.getPerspectiveTransform(cornerI,cornerT)
#                 M = cv2.getAffineTransform(cornerI,cornerT)
                M[0:2,0:2] = M[0:2,0:2]/np.linalg.norm(M[0:2,0:2],ord=2)
                corner = np.float64(corner)
                cornerIT = cv2.perspectiveTransform(corner.reshape(-1,1,2),M)
#                 cornerIT.reshape(-1,2)
                frame2 = cv2.warpPerspective(framegray,M,(VWIDTH,VHIGH))
#                 for i in range(4):
#                     frame2[cornerIT[i][0][1]-3:cornerIT[i][0][1]+3,cornerIT[i][0][0]-3:cornerIT[i][0][0]+3,:] = [255,0,255]
                
#                 frame2 = cv2.warpAffine(frame,M,(VWIDTH*2,VHIGH*2))
#             logoContourFlatten = logoContour.flatten()1
#             logoContourX = logoContourFlatten[0:-1:2]
#             logoContourY = logoContourFlatten[1:-1:2]
#             pp1 = logoContour[logoContourX.argmax()]
#             pp2 = logoContour[logoContourX.argmin()]
#             pp3 = logoContour[logoContourY.argmax()]
#             pp4 = logoContour[logoContourY.argmin()]
#             logoContourPoint = np.array([pp1,pp2,pp3,pp4])
#             cv2.drawContours(frame, logoContourPoint, -1, (255,255), 2)

###########################################################################
###########################################################################
#             rect = cv2.minAreaRect(logoContour)
#             box = np.int0(cv2.boxPoints(rect))
#             xaxis = np.array([box[0,0],box[1,0],box[2,0],box[3,0]])
#             yaxis = np.array([box[0,1],box[1,1],box[2,1],box[3,1]])
#             cropst = np.array([yaxis.min()-10,xaxis.min()-10])
#             croped = np.array([yaxis.max()+10,xaxis.max()+10])
#             crop = framegray[cropst[0]:croped[0],cropst[1]:croped[1]]
# #             crop = cv2.resize(crop,(256,256))
#             maskNew = np.zeros([VHIGH,VWIDTH],np.uint8)
#             maskNew[cropst[0]:croped[0],cropst[1]:croped[1]] = 255
# #             framgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#             kpdst, desdst = kaze.detectAndCompute(crop, None)
#             matches = flann.knnMatch(dessrc, desdst, k=2)
#             matchesMask = [[0,0] for i in range(len(matches))]
#             goodMatches = []
#             for i,(m,n) in enumerate(matches):
#                 if m.distance < 0.7*n.distance:
#                     matchesMask[i]=[1,0]
#                     goodMatches.append(m)
#              
#             MIN_MATCH_COUNT = 10
#             if len(goodMatches) > MIN_MATCH_COUNT:
#                 # 获取关键点的坐标
#                 src_pts = np.float64([ kpsrc[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
#                 dst_pts = np.float64([ kpdst[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
#                 M, maskNewNew = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
# #                 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
# #                 M, mask = cv2.findHomography(dst_pts, src_pts, cv2.LMEDS)
# #                 M[0:2,0:2] = M[0:2,0:2]/np.linalg.norm(M[0:2,0:2],ord=2)
# # #                 
# #                 frame = cv2.warpPerspective(frame,M,(VWIDTH,VHIGH))
# #                 print(M)
#                 # 获得原图像的高和宽
#                 h,w = logoimggray.shape
# #                 
#                 # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
#                 pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#                 dst = cv2.perspectiveTransform(pts,M)
#                 # 原图像为灰度图
#                 cv2.polylines(frame,[np.int32(dst)],True,[255,255,255],2, cv2.LINE_AA)
###########################################################################
###########################################################################


#             blankimg = np.float32(blankimg)
#             dst = cv2.cornerHarris(blankimg,2,3,0.04)
#             dst = cv2.dilate(dst,None)
#             frame[dst>0.1*dst.max()]=[0,255,0]
#             cv2.floodFill(blankimg,maskNew,(0,0),255)
#             print(logoContour)
            
#         else:
#             continue
#         if(len(contours)>0):
#             cv2.drawContours(frame, contours, -1, (0,255,0), 2)
        # get the logo conture from image
        
        
#         cv2.imshow('frame',frame)
        cv2.imshow('blankimg',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
        