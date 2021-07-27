import torch 
import  cv2
import torchvision
import os 
def yolo(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 
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
# img = cv2.imread('./data-mutilmodel/h1.jpg')
# corr_data = yolo(img)
# red_corr = corr_data[corr_data['name'] == 'red']
# xcenter_pixel = red_corr['xcenter'].values[0]-600
# ycenter_pixel = red_corr['ycenter'].values[0]

# # 67pixel = 3cm 
# xcenter_mm = xcenter_pixel/62*30
# ycenter_mm = ycenter_pixel/62*30
# print(xcenter_mm)
# print (ycenter_mm)
# cv2.imshow('aaa', img)
# cv2.waitKey()
# images = []
file = open('corr.txt', 'w')
for file_name in os.listdir('./data-mutilmodel'):
    split1 = file_name.split('(')
    name = split1[0]
    split2 = split1[1].split(')')
    split3 = split2[0].split(',')
    x = split3[0]
    y = split3[1]
    
    img = cv2.imread(os.path.join("./data-mutilmodel", file_name))
    corr_data = yolo(img)
    red_corr = corr_data[corr_data['name'] == 'red']
    xcenter_pixel = red_corr['xcenter'].values[0]-600
    ycenter_pixel = red_corr['ycenter'].values[0]
    #67pixel = 3cm 
    xcenter_mm = xcenter_pixel/62*30
    ycenter_mm = ycenter_pixel/62*30

    err_x = (int(x)-xcenter_mm)*(int(x)-xcenter_mm)
    err_y = (int(y)-ycenter_mm) * (int(y)-ycenter_mm)
    import math
    err = math.sqrt(err_x+err_y)

   
    file.writelines([name," ", str(x), " ", str(y), " ", str(xcenter_mm)," ", str(ycenter_mm)," ", str(err_x), " ", str(err_y)," ", err "\n"])
    # print(file_name.split('(')[0])
    