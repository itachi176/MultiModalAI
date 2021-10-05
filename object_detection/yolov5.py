import torch 
import  cv2
import torchvision
import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import ChessboardCalibration_master.tools.predict as predict 
import time
def yolo(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/super-best.pt')
    model.conf = 0.6
    model.iou = 0.6
    img1 = image[..., ::-1]
    results = model(img1)
    # results.print()
    # results.show()
    # results.save()
    results.xywh[0]
    a = results.pandas().xywh[0]
    print(a)
    # results.show()
    # print(a['confidence'][0])
    boxes = []
    return a 
# x = yolo()
# y = x[x['name']=='red']
# print(y['xcenter'].values[0])
# img = cv2.imread('./data/dataset_15082021/a.jpg')
# # # predict.pred(754.3876342773438, 546.3013916015625, "./a.jpg")
# corr_data = yolo(img)
# # print(corr_data)
# red_corr = corr_data[corr_data['name'] == 'red']
# # print(red_corr["xcenter"].values[1])
# xcenter_pixel = red_corr['xcenter'].values
# print(len(red_corr))
# ycenter_pixel = red_corr['ycenter'].values[0]-28

# # # 67pixel = 3cm 
# xcenter_mm = xcenter_pixel/62*30
# ycenter_mm = ycenter_pixel/62*30
# print(xcenter_mm)
# # print (ycenter_mm)
# cv2.imshow('aaa', img)
# cv2.waitKey()
# # images = []
file = open('test_corr.txt', 'w')
count = 0
for file_name in os.listdir('./data/dataset_15082021'):
    print(file_name)
    split1 = file_name.split('(')
    name = split1[0]
    split2 = split1[1].split(')')
    split3 = split2[0].split(',')
    print(split3)
    x = split3[0]
    y = split3[1]
    img = cv2.imread(os.path.join("./data/dataset_15082021", file_name))
    start1 = time.time()
    corr_data = yolo(img)
    end1 = time.time()
    time_detect = end1-start1
    start = time.time()
    red_corr = corr_data[corr_data['name'] == 'red']
    xcenter_pixel = red_corr['xcenter'].values[0]
    ycenter_pixel = red_corr['ycenter'].values[0]
    #67pixel = 3cm 
    # # time.sleep(2)
   
    xcenter_mm, ycenter_mm = predict.pred(xcenter_pixel, ycenter_pixel, os.path.join("./data/dataset_15082021", file_name))
    # xcenter_mm = xcenter_pixel/62*30
    xcenter_mm = round(xcenter_mm,2)*10
    # ycenter_mm = ycenter_pixel/62*30
    ycenter_mm = round(ycenter_mm, 2)*10

    err_x = abs(int(x)-xcenter_mm)
    err_x = round(err_x,2)
    err_y = abs(int(y)-ycenter_mm) 
    err_y = round(err_y,2)
    import math
    err = math.sqrt(err_x*err_x+err_y*err_y)
    err = round(err, 2)
    w = red_corr['width'].values[0]
    # w = round(w,2)
    h = red_corr['height'].values[0]
    end = time.time()
    time_cal = end - start
    # h = round(h, 2)
    print(name)
    file.writelines([str(time_detect)," ", str(time_cal), "\n"])
    # file.writelines([name, " ", str(xcenter_pixel), " ", str(ycenter_pixel), " ", str(w), " ", str(h), "\n"])
 
# # #     # print(file_name.split('(')[0])
# # #     # file.writelines([file_name, "       ",str(xcenter_mm), "        ", str(ycenter_mm),"\n"])
    