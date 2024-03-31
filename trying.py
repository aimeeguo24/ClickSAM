import numpy as np
import cv2
import os
image=cv2.imread('data/BUSI2Dtrain/labels/benign (1)_mask.png')
###binarising
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

###applying morphological operations to dilate the image
kernel=np.ones((3,3),np.uint8)
dilated=cv2.dilate(th2,kernel,iterations=3)

### finding contours, can use connectedcomponents aswell
contours = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

### converting to bounding boxes from polygon
contours=[cv2.boundingRect(cnt) for cnt in contours]
### drawing rectangle for each contour for visualising
for cnt in contours:
    x,y,w,h=cnt
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)