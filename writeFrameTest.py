import numpy as np
import cv2
# VWIDTH = 1280
# VHIGH = 720
VWIDTH = 640
VHIGH = 480
cap = cv2.VideoCapture(0)
ret = cap.set(3,VWIDTH)
ret = cap.set(4,VHIGH)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter()
succes = out.open('output.mp4v',fourcc, 15.0, (VWIDTH,VHIGH),True)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
#         frame = cv2.flip(frame,0)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()