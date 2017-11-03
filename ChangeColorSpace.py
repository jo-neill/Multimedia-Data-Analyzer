import cv2
import numpy

im = cv2.imread("unfiltered.jpg")
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

cv2.imshow("hsv image", im_hsv)

blue = numpy.uint8([[[0,0,255]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsv_blue)