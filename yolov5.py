import torch 
import  cv2
import torchvision
def yolo():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 
    img1 = cv2.imread('./test-obj/y1.jpg')[..., ::-1]
    results = model(img1)
    results.print()  
    # results.show()
    results.save()
    results.xywh[0]
    a = results.pandas().xywh[0]
    print(a)
    results.show()
    print(a['confidence'][0])
    boxes = []
    return a 
# x = yolo()
# y = x[x['name']=='red']
# print(y['xcenter'].values[0])
# img = cv2.imread('./test-obj/y1.jpg')
# cv2.imshow('aaa', img)
# cv2.waitKey()