import torch 
import  cv2
import os 
import ChessboardCalibration_master.tools.predict as predict #calibration 
import time
from text_to_speech import *
from text_to_speech import *
from threading import Thread 
from similary import *

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

def speak():
    a = ''
    global flag
    flag = 0
    while(a != "tạm biệt\n"):
        a = speech_to_text()
        try:
            a = similary(a)
        except:
            print('loi roi!!!')
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
            flag = 1
            print("flag:",flag)
            
def video():
    global flag 
    # cap = cv2.VideoCapture(0)
    # while(True):
    #     # flag = 0
    #     ret, frame = cap.read()
    #     if ret == False:
    #         break
    #     k = cv2.waitKey(1)
        
    #     cv2.imshow("hi", frame)
    #     # a = input()
    #     # if 0xFF == ord('c'):
        
    #     # if k%256 == 27:
    #     #     # ESC pressed
    #     #     cv2.imwrite("./robot_cam/image.jpg", cv2.imread('./a.jpg'))
    #     #     str = get_corr(cv2.imread('./a.jpg'))
    #     #     print(str)
    #     #     print("hello")
    #     if flag == 1:
    #         cv2.imwrite("./robot_cam/image.jpg", cv2.imread('./a.jpg'))
    #         str = get_corr(cv2.imread('./a.jpg'))
    #         print(str)
    #         print("hello")
    #     flag = 0
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break


    # cap.release()
    # cv2.destroyAllWindows()

th2 = Thread(target=speak)
th2.start()
th1 = Thread(target=video)
th1.start()

# img = cv2.imread('./a.jpg')
# cv2.imshow('aaa', img)
# cv2.waitKey()