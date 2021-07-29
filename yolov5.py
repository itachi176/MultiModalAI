from test import CONFIDENCE_THRESHOLD
import torch 
import  cv2
import torchvision
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
# x = yolo()
# y = x[x['name']=='red']
# print(y['xcenter'].values[0])
# img = cv2.imread('./anhnam/a.jpg')
# predict.pred(754.3876342773438, 546.3013916015625, "./a.jpg")
# corr_data = yolo(img)
# red_corr = corr_data[corr_data['name'] == 'red']
# xcenter_pixel = red_corr['xcenter'].values[0]-609
# ycenter_pixel = red_corr['ycenter'].values[0]-28

# # 67pixel = 3cm 
# xcenter_mm = xcenter_pixel/62*30
# ycenter_mm = ycenter_pixel/62*30
# print(xcenter_mm)
# # print (ycenter_mm)
# cv2.imshow('aaa', img)
# cv2.waitKey()
# images = []
file = open('test_corr.txt', 'w')
for file_name in os.listdir('./dataset_29072021'):
    split1 = file_name.split('(')
    name = split1[0]
    split2 = split1[1].split(')')
    split3 = split2[0].split(',')
    print(split3)
    x = split3[0]
    y = split3[1]
    
    img = cv2.imread(os.path.join("./dataset_29072021", file_name))
    corr_data = yolo(img)
   
    red_corr = corr_data[corr_data['name'] == 'red']
    xcenter_pixel = red_corr['xcenter'].values[0]-625
    ycenter_pixel = red_corr['ycenter'].values[0]-34
    #67pixel = 3cm 
    # time.sleep(2)
    # xcenter_mm, ycenter_mm = predict.pred(xcenter_pixel, ycenter_pixel, os.path.join("./dataset_29072021", file_name))
    xcenter_mm = xcenter_pixel/62*30
    xcenter_mm = round(xcenter_mm,2)
    ycenter_mm = ycenter_pixel/62*30
    ycenter_mm = round(ycenter_mm, 2)

    err_x = abs(int(x)-xcenter_mm)
    err_x = round(err_x,2)
    err_y = abs(int(y)-ycenter_mm) 
    err_y = round(err_y,2)
    import math
    err = math.sqrt(err_x*err_x+err_y*err_y)
    err = round(err, 2)
    w = red_corr['width'].values[0]/62*30
    w = round(err,2)
    h = red_corr['height'].values[0]/62*30
    h = round(err, 2)

    file.writelines([name," ", str(err_x), " ", str(err_y), " ", str(err),"\n"])
    
# #     # print(file_name.split('(')[0])
# #     # file.writelines([file_name, "       ",str(xcenter_mm), "        ", str(ycenter_mm),"\n"])
    