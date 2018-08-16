__author__ = 'Lime'
import cv2
import numpy as np
from sklearn.cluster import KMeans
#img = cv2.imread('images/abudefduf vaigiensis/sample287.png')
img = cv2.imread('images/phosphorus/IMG_0731.jpg')
#resImg = cv2.resize(img, None, fx=0.2 , fy=0.2, interpolation=cv2.INTER_CUBIC )


new_size=500
h, w = img.shape[0], img.shape[1]
ds_factor = new_size / float(h)

if w < h:
    ds_factor = new_size / float(w)

new_size = (int(w * ds_factor), int(h * ds_factor))
resImg = cv2.resize(img, new_size)


detector = cv2.FeatureDetector_create("Dense")
detector.setInt("initXyStep", 20)
detector.setInt("initFeatureScale", 40)
detector.setInt("initImgBound", 4)
kp = detector.detect(resImg, None)

gray_image = cv2.cvtColor(resImg, cv2.COLOR_BGR2GRAY)
kp, des = cv2.SIFT().compute(gray_image, kp)

img2 = cv2.drawKeypoints(resImg,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Original', resImg)
cv2.imshow('Dense', img2)
cv2.waitKey()