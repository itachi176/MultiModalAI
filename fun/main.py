import cv2
import numpy as np
import pytesseract


# Load image, convert to HSV format, define lower/upper ranges, and perform
# color segmentation to create a binary mask
# a = str(input())
image = cv2.imread('../data/a.png', 0)

thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
blur = cv2.GaussianBlur(thresh, (7,7), 0)
cnts = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    print(pytesseract.image_to_string(image[y:y+h, x:x+w], lang = 'eng'))
    # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    # cv2.imshow('a', image[y:y+h, x:x+w])
    # cv2.waitKey()

cv2.imshow('result', image)
cv2.waitKey()