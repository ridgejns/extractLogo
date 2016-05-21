""" This is a class for axtract the position and image out of the Lego box,
    based on the Lego logo parts.

    Author:  Li Sun & Lyu Yaopengfei
    Date: 16-May-2016
"""
# import the necessary packages
import numpy as np
import cv2
import imgPreprocessing


class Lego(object):
    def __init__(self, image):
        # initialize attributes
        self._pureLogo = cv2.imread('./img/purelogo256.png')
        self._image = image
        self._rotated_image = image
        self._rotate_angle = None
        self._information = np.zeros([100,100,3],dtype=np.uint8)
        self._logo = np.zeros([100,100,3],dtype=np.uint8)
        self._logo_box = None

        # Parameters
        self.MIN_MATCH_COUNT = 6
        self.MAX_MATCH_COUNT = 12

        # Flags
        self._has_rotated_image = False
        self._has_rotate_angle = False
        self._has_potential_logo = False
        self._has_valid_logo = False
        self._has_information = False

        #the main part of the algorithm
        self._logo_detect()
        self._get_rotate_angle()
        self._get_rotated_image()
        self._get_information()

    def _logo_detect(self):
        rows, cols, _ = self._image.shape

        # Convert BGR to HSV
        hsv = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)

        # define range of red color in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])

        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([179, 255, 255])

        # Threshold the HSV image to get only red colors
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        self._mask = mask1 + mask2

        _, contours, _ = cv2.findContours(self._mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        new_contours = []
        for idx, contour in enumerate(contours):
            # only take the first five of the contours
            if idx >= 5:
                break
            # generate the area of the found contour
            area = cv2.contourArea(contour)
            # the perimeter
            perimeter = cv2.arcLength(contour, True)
            # check if the contour is a square by formula below:
            if (np.sqrt(area) * 4 <= perimeter * 1.2) & (np.sqrt(area) * 4 >= perimeter * 0.8):
                new_contours.append(contour)

        if (len(new_contours) >= 1):
            # take the biggest suitable contour
            cnt = sorted(new_contours, key=cv2.contourArea, reverse=True)[0]

            # compute the rotated bounding box of the contour
            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))
            ylimit, xlimit, _ = self._image.shape

            xaxis = np.array([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])
            yaxis = np.array([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])
            cropst = np.array([yaxis.min()-10, xaxis.min()-10])
            croped = np.array([yaxis.max()+10, xaxis.max()+10])

            if (cropst[0]>0) & (cropst[1]>0) & (croped[0]<ylimit) & (croped[1]<xlimit):
                # crop the logo area around the logo box
                self._logo = self._image[cropst[0]:croped[0], cropst[1]:croped[1]]
                self._logo_box = box
                self._has_potential_logo = True

    def _get_rotate_angle(self):
        if self._has_potential_logo:
            akaze = cv2.AKAZE_create()

            gray_image1 = cv2.cvtColor(self._pureLogo, cv2.COLOR_BGR2GRAY)
            gray_image2 = cv2.cvtColor(self._logo, cv2.COLOR_BGR2GRAY)

            kp1, des1 = akaze.detectAndCompute(gray_image1, None)
            kp2, des2 = akaze.detectAndCompute(gray_image2, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            if des2 is not None:
                matches = bf.knnMatch(des1, des2, k=2)
            else:
                matches = []

            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.7*n.distance:
                        good_matches.append(m)
            except:
                pass

            # the key-points matched within certain ranges.
            if (len(good_matches) >= self.MIN_MATCH_COUNT) & (len(good_matches) >= self.MAX_MATCH_COUNT):
                src_pts = np.float64([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float64([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # calculate the rotate angles
                lyu = imgPreprocessing.LogoAffinePos(self._pureLogo)
                Ang = lyu.__calcuAngle(src_pts, dst_pts)
                # check if there is a angle(if the potential logo is a valid lego logo )
                if np.isnan(Ang) | (len(good_matches) < 6):
                    self._has_valid_logo =False
                else:
                    self._rotate_angle = Ang/np.pi*180
                    self._has_rotate_angle = True
                    self._has_valid_logo = True

                    # im3 = cv2.drawMatchesKnn(self._pureLogo, kp1, self._logo, kp2, good_matches, None, flags=2)
                    # cv2.imshow("AKAZE matching", im3)
                    # cv2.waitKey(0)

    def _get_information(self):
        if self._has_valid_logo & self._has_rotated_image:
            ylimit, xlimit, _ = self._image.shape
            height, weight, _ = self._logo.shape

            cropst = np.array([self._logo_center_y+height/2, self._logo_center_x-weight/2])
            croped = np.array([self._logo_center_y+height+height/2, self._logo_center_x+weight/2])

            if (cropst[0]>0) & (cropst[1]>0) & (croped[0]<ylimit) & (croped[1]<xlimit):
                img = self._rotated_image[cropst[0]:croped[0], cropst[1]:croped[1]]

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                kernel = np.ones((2,2),np.uint8)
                thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
                self._information = thresh
                self._has_information = True

    def _get_rotated_image(self):
        if self._has_valid_logo & self._has_rotate_angle:
            imgH, imgW, _ = self._image.shape
            box = self._logo_box
            xaxis = np.array([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])
            yaxis = np.array([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])

            # calculate the center of the lego logo
            self._logo_center_x = int(round(np.mean(xaxis),0))
            self._logo_center_y = int(round(np.mean(yaxis),0))
            M = cv2.getRotationMatrix2D((self._logo_center_x, self._logo_center_y), self._rotate_angle, 1)
            self._rotated_image = cv2.warpAffine(self._image, M, (imgW, imgH))
            self._has_rotated_image = True

    def get_logo_box(self):
        return self._logo_box

    def get_logo_image(self):
        return self._logo

    def get_information_part(self):
        return self._information

    def get_rotated_image(self):
        return self._rotated_image
