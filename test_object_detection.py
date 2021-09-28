from object_detection.yolov5 import *
import cv2 
img = cv2.imread('./data/dataset_15082021/a.jpg')
img = img[..., ::-1]
cv2.imshow("ssss", img)
cv2.waitKey()
