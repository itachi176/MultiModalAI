import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('./gesture/ml.model')
import cv2 
img = cv2.imread('./test.png', 0)
img = cv2.resize(img, (28, 28))
a = tf.expand_dims(img, axis=0)
x = model.predict(a)
y = np.argmax(x)
print(y)
cv2.imshow('aaa', img)
cv2.waitKey()