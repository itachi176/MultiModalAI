from __future__ import print_function
from tensorflow import keras 
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle
import numpy as np


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

model.save("ml.model")

print("model saved")