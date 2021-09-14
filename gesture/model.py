from __future__ import print_function
from tensorflow import keras 
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
import cv2

(X_train, y_train), (X_test, y_test) = mnist.load_data()
pant = mnist.load_data()

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

X_train = X_train/255.0
X_test = X_test/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation ="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3)

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test loss is " + str(test_loss))
print("Test acc is " + str(test_acc * 100))
prediction = model.predict(X_test[:5])

num = 0

for i in range (len(prediction)):
    guess = np.argmax(prediction[i])
    actual = y_test[i]
    print("The computer guessed that the number was a ", guess)
    print("The number was actually a ", actual)
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    plt.show()
img = cv2.imread('./test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, imgInv = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
img = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
a = tf.expand_dims(img, axis=0)
x = model.predict(a)
y = np.argmax(x)
print(y)
model.save("ml.model")

# print("model saved")