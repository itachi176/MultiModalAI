# import necessary packages

import cv2
import sys
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import serial
import time
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import ChessboardCalibration_master.tools.predict as predict
import object_detection

def yolo(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/super-best.pt')
    model.conf = 0.6
    model.iou = 0.6
    img1 = image[..., ::-1]
    results = model(img1)
    results.print()  
    # results.show()
    # results.save()
    results.xywh[0]
    a = results.pandas().xywh[0]
    # print(a)
    results.show()
    # print(a['confidence'][0])
    boxes = []
    return a 


def get_corr(img):
    # img = cv2.imread(img_path)
    corr_data = yolo(img)
    red_corr = corr_data[corr_data['name'] == 'red']
    xcenter_pixel = red_corr['xcenter'].values[0]
    ycenter_pixel = red_corr['ycenter'].values[0]
    #67pixel = 3cm 
    # time.sleep(2)
    xcenter_mm, ycenter_mm = predict.pred(xcenter_pixel, ycenter_pixel, "./robot_cam/image.jpg")
    # xcenter_mm = xcenter_pixel/62*30
    xcenter_mm = round(xcenter_mm,2)*10
    # ycenter_mm = ycenter_pixel/62*30
    ycenter_mm = round(ycenter_mm, 2)*10
    # err_x = abs(int(x)-xcenter_mm)
    # err_x = round(err_x,2)
    # err_y = abs(int(y)-ycenter_mm) 
    # err_y = round(err_y,2)
    import math
    # err = math.sqrt(err_x*err_x+err_y*err_y)
    # err = round(err, 2)
    return "{} {}".format(xcenter_mm, ycenter_mm)

# ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)

def write_data(string):
    ser.write(string.encode())
    ser.close()

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('./gesture/mp_hand_gesture')

# Load class names
f = open('./gesture/gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret1, frame = cap1.read()
    ret2, img = cap2.read()
    if ret1:
        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)
        
        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0,0,255), 2, cv2.LINE_AA)
        if cv2.waitKey(33) == ord("a"):
            if (className == "004"):
                print("hi")
                # cv2.imwrite("./robot_cam/image.jpg", cv2.imread('./a.jpg'))
                cv2.imwrite("./robot_cam/image.jpg", img)
    
                str = get_corr(img)
                print(str)
            # arduino.write_data(className)
            else:
                print("gui di: ", className)
                write_data(className)


        # Show the final output
        cv2.imshow("human_cam", frame) 
    if ret2:
        cv2.imshow("object_cam", img)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap1.release()
cap2.release()

cv2.destroyAllWindows()