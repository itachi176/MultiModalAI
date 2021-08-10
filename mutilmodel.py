from speech_to_text import speech_to_text
from threading import Thread
import cv2
import time
from text_to_speech import *
from test import *
from price import *
import pandas as pd
from similary import *

INPUT_FILE = 'e.png'
OUTPUT_FILE = 'predicted.jpg'
LABELS_FILE = 'yolo-coco/coco.names'
CONFIG_FILE = 'yolo-coco/yolov3.cfg'
WEIGHTS_FILE = 'yolo-coco/yolov3.weights'
CONFIDENCE_THRESHOLD = 0.7
import cv2
import pickle
from  preprocess import *
import numpy as np
import cv2
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from yolov5 import *
data = my_data()
# data = shuffle(data)
train = data 
test = data[:20]
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
print(X_train.shape)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
print(X_test.shape)
y_test = [i[1] for i in test]

from tensorflow.python.framework import ops
ops.reset_default_graph()
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
# 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.fit(X_train, y_train, n_epoch=10)


def obj_det():
    LABELS = open(LABELS_FILE).read().strip().split("\n")

    np.random.seed(4)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

    image = cv2.imread(INPUT_FILE)
    image = cv2.resize(image, (840, 640))
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE_THRESHOLD:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                            CONFIDENCE_THRESHOLD)

    label_pred = []
    corr = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            text = "{}:{:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # if (LABELS[classIDs[i]] != 'person'):
        #     (x_object, y_object) = (boxes[i][0], boxes[i][1])
        #     (w_object, h_object) = (boxes[i][2], boxes[i][3])
        #     cv2.rectangle(image, (x_object, y_object), (x_object+w_object, y_object+h_object), color, 2)
        #     text = "{}:{:.4f}".format(LABELS[classIDs[i]], confidences[i])
        #     cv2.putText(image, text, (x_object, y_object-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            name = LABELS[classIDs[i]]
            label_pred.append(name)
            corr.append([x,y,w,h])
        # print('d' in label_pred)
    import matplotlib.pyplot as plt   
    plt.imshow(image)
    plt.show()
    return label_pred, corr



# def video():
#     cap = cv2.VideoCapture(0)

#     while(True):
        
#         ret, frame = cap.read()
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
# mylabel = ""
def video():
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    temp = 0
    while(True):
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(frame, 1.3, 5)
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h),(255,0,0), 1)
            new = frame[y:y+h, x:x+w]
            new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
            new = cv2.resize(new, (50,50))
            new = new.reshape(50,50,1)
            result = model.predict([new])[0]
            global my_label
            if np.argmax(result) == 0:
                my_label = 'hoang'
            if np.argmax(result) == 1:
                my_label = 'mytam'
            else:
                mylabel = 'ronaldo'
            cv2.putText(frame, '{}'.format(my_label), (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

           
        cv2.imshow("frame", frame)
        if cv2.waitKey(100) & 0xFF==ord('q'):
            break

    cv2.destroyAllWindows()


df = pd.read_csv("./data.csv")
user = pd.read_csv('./user.csv')

def price_object(name):
    price = df[df['Name'] == name]['price'].values[0]
    return price


def color_object(name):
    color = df[df['Name'] == name]['color'].values[0]
    return color

def speak():
    a = ''
    while(a != "tạm biệt\n"):
        a = speech_to_text()
        a = similary(a)
        print(a)
        if a == "lấy quả táo trên bàn\n":
            label, corr = obj_det()
            if ("apple" in label):
                print('toa do qua tao la: ', corr[label.index('apple')-1])
            else:
                print('khong co qua tao nao tren ban')
        if a == "xin chào\n":
            text_to_speech("chào bạn, tôi có thể giúp gì cho bạn")
        if a == "đây là gì":
            text_to_speech("đây là {}".format(name))
        if a == "giá bao nhiêu":
            price = price_object(name)
            print(price)
            text_to_speech("giá của {} là {}".format(name, price))
        if a == "màu gì":
            color = color_object(name)
            print(color)
            text_to_speech("màu của {} là {}".format(name, color))
        if a == "tạm biệt\n":
            text_to_speech("Hẹn gặp lại hoàng")
        if a == "đây là ai\n":
            text_to_speech('đây là {}'.format(my_label))
        if a == "giới tính là gì\n":
            text_to_speech("giới tính là {}".format(user[user['name'] == my_label]['gioi tinh'].values[0]))
        if a == "mã số sinh viên là bao nhiêu\n":
            mssv = str(user[user['name']==my_label]['id'][0])
            text_to_speech("mã số sinh viên là {}".format(mssv))
            print(mssv)
        if a == "nghề nghiệp là gì\n":
            job = user[user['name']==my_label]['nghe nghiep'].values[0]
            text_to_speech("nghề nghiệp của {} là {}".format(my_label, job))
            print('job')
        if a == "học lớp nào\n":
            text_to_speech("{} học lớp {}".format(my_label, user[user['name']==my_label]['lop'][0]))
        if a == "lấy cho tôi vật vào hộp màu đỏ\n":
            corr_data = yolo()
            red_corr = corr_data[corr_data['name'] == 'red']
            xcenter_pixel = red_corr['xcenter'].values[0]
            ycenter_pixel = red_corr['ycenter'].values[0]
            #67pixel = 3cm 
            xcenter_mm = xcenter_pixel/67*30
            ycenter_mm = ycenter_pixel/67*30
            print("x : {}, y: {}".format(xcenter_mm, ycenter_mm)) 
            


# th1 = Thread(target=video)
# th1.start()
th1 = Thread(target=video)
th1.start()
th2 = Thread(target=speak)
th2.start()
