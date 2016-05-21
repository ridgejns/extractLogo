import numpy as np
import cv2

def cvShowManyImages(title, showSize = 'L', *args):
    
    if (showSize is 'S'):
        s = [480, 640]
    elif (showSize is 'M'):
        s = [600, 800]
    elif (showSize is 'L'):
        s = [768, 1024]
    elif (showSize is 'xL'):
        s = [1050,1400]
    elif (showSize is 'xxL'):
        s = [1200,1600]
    else:
        s = [480, 640]
    
    nargs = len(args)
    if (nargs is 0):
        print("Number of arguments too small....\n")
        return
    elif ((nargs is 1)):
        layout = [1,1]
        h,w = [3,4]
        sH = s[0]
        sW = s[1]
        sEach = [sH,sW]
    elif ((nargs is 2)):
        layout = [2,1]
        h,w = [4,3]
        sH = s[1]
        sW = s[0]
        sEach = [sH/2,sH/2/3*4]
    elif ((nargs is 3) | (nargs is 4)):
        layout = [2,2]
        h,w = [3,4]
        sH = s[0]
        sW = s[1]
        sEach = [sH/2,sW/2]
    elif ((nargs is 5) | (nargs is 6)):
        layout = [3,2]
        h,w = [4,3]
        sH = s[1]
        sW = s[0]
        sEach = [sW/2/4*3*sW/2]
    elif ((nargs is 7) | (nargs is 8) | (nargs is 9)):
        layout = [3,3]
        h,w = [3,4]
        sEach = [sH/3,sW/3]
#     elif ((nargs is 10) | (nargs is 11) | (nargs is 12)):
    else:
        print("Number of arguments too large....\n")
        return
    
    mImg = np.uint8(sH,sW,3)
#     for idx, img in enumerate(args):
#         mImg[]
    
if __name__ == '__main__':
    img1 = cv2.imread('/Users/lvypf/OneDrive/myprjs/pythonSpace/extractText/img/logorot180.png')
    img2 = cv2.imread('/Users/lvypf/OneDrive/myprjs/pythonSpace/extractText/img/logorot25.png')
    
    cv2.namedWindow('image1')
    cv2.namedWindow('image2')
    
    while(1):
        cv2.imshow('image1',img1)
#         cv2.imshow('image2',img2)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()