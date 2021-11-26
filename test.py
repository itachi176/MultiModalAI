import torch
import cv2
import time 

def yolo(image):
    model = torch.hub.load('./yolov5', 'custom', path='./model/super-best.pt', source='local')
    model.conf = 0.6
    model.iou = 0.6
    img1 = image[..., ::-1]
    results = model(img1)
    # results.print()  
    # results.show()
    # results.save()
    # results.xywh[0]
    a = results.pandas().xywh[0]
    # print(a)
    # results.show()
    # print(a['confidence'][0])
    boxes = []
    return a 
start = time.time()
a = yolo(cv2.imread('a.jpg'))
end = time.time()
print("time:", end- start)