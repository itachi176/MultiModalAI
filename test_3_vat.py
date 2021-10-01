import torch 
import  cv2
import os 
import ChessboardCalibration_master.tools.predict as predict #calibration 
import time
from speech.text_to_speech import *

from threading import Thread 
from speech.similary import *

def yolo(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/super-best.pt')
    model.conf = 0.6
    model.iou = 0.6
    img1 = image[..., ::-1]
    results = model(img1)
    results.print()  
    results.show()
    # results.save()
    results.xywh[0]
    a = results.pandas().xywh[0]
    # print(a)
    # results.show()
    # print(a['confidence'][0])
    boxes = []
    return a 

while(True):
    x = speech_to_text()
    try:
        x= similary(x)
        print(x)
    except:
        pass
    
    if x == "lấy cái hộp\n":
        print('red')
    if x =="lấy cái vòng tròn\n":
        print("green")
    if x == "lấy cục pin\n":
        print("yellow")
    