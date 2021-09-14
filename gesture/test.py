import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('./gesture/ml.h5')
import cv2 
def pred(img):
# img = cv2.imread('./test4.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
    gray_img = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (28, 28))
    # img = np.pad(img, (10,10), 'constant', constant_values=0 )
    # img = cv2.resize(img, (28,28))/255
    # a = tf.expand_dims(img, axis=0)
    # x = model.predict(a)
    # y = np.argmax(x)
    # print(y)
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sắp xếp các contour theo diện tích giảm dần:
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_sort = np.argsort(area_cnt)[::-1]
    # Top 20 contour có diện tích lớn nhất
    print("area:", area_sort)
    def _drawBoundingBox(img, cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        new_img = img[y:y+h, x:x+w]
        return new_img
    new_img = _drawBoundingBox(img, contours[1])
    new_img= cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.resize(new_img, (28, 28))
    new_img = np.pad(new_img, (10,10), 'constant', constant_values=0 )
    new_img = cv2.resize(new_img, (28,28))/255
    print(new_img.shape)
    a = tf.expand_dims(new_img, axis=0)
    x = model.predict(a)
    y = np.argmax(x)
    print(y)
    return y
# img = cv2.imread('./test4.png')
# print(pred(img))



# print(img.shape)
# cv2.imshow('aaa', new_img)
# cv2.waitKey()