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
    return label_pred, corr
    # cv2.imshow("Image", image)
    # cv2.waitKey()


def video():
    cap = cv2.VideoCapture(0)

    while(True):
        
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


df = pd.read_csv("./data.csv")


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
                print('toa do qua tao la: ', label.index('apple'))
            else:
                print('khong co qua tao nao tren ban')
        if a == "xin chào\n":
            text_to_speech("chào hoàng, tôi có thể giúp gì cho bạn")
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


# th1 = Thread(target=video)
# th1.start()
th1 = Thread(target=video)
th1.start()
th2 = Thread(target=speak)
th2.start()
