import torch 
import  cv2
import os 
import ChessboardCalibration_master.tools.predict as predict #calibration 
import time

def yolo(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='super-best.pt')
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
    # results.show()
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
    
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    if ret == False:
        break
    k = cv2.waitKey(1)
    
    cv2.imshow("hi", frame)
    # a = input()
    # if 0xFF == ord('c'):
    
    if k%256 == 27:
        # ESC pressed
        cv2.imwrite("./robot_cam/image.jpg", cv2.imread('./a.jpg'))
        str = get_corr(cv2.imread('./a.jpg'))
        print(str)
        print("hello")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
# img = cv2.imread('./a.jpg')
# cv2.imshow('aaa', img)
# cv2.waitKey()