import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('./gesture/ml.h5')
import cv2 
img = cv2.imread('./test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, imgInv = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
# img = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
img = np.pad(img, (10,10), 'constant', constant_values=0 )
img = cv2.resize(img, (28,28))/255
a = tf.expand_dims(img, axis=0)
x = model.predict(a)
y = np.argmax(x)
print(y)
print(img.shape)
cv2.imshow('aaa', img)
cv2.waitKey()