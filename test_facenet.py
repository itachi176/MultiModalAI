import os 
import cv2
import numpy as np 
import mtcnn 

path = './pretrain_model'
a=  os.path.join(path, "nn4.small2.v1.t7")
def load_torch(path):
    model = cv2.dnn.readNetFromTorch(path)
    return model
encoder = load_torch(a)

#blob image 
def _blobImage(image, out_size = (300, 300), scaleFactor = 1.0, mean = (104.0, 177.0, 123.0)):
  """
  input:
    image: ma trận RGB của ảnh input
    out_size: kích thước ảnh blob
  return:
    imageBlob: ảnh blob
  """
  # Chuyển sang blobImage để tránh ảnh bị nhiễu sáng
  imageBlob = cv2.dnn.blobFromImage(image, 
                                    scalefactor=scaleFactor,   # Scale image
                                    size=out_size,  # Output shape
                                    mean=mean,  # Trung bình kênh theo RGB
                                    swapRB=False,  # Trường hợp ảnh là BGR thì set bằng True để chuyển qua RGB
                                    crop=False)
  return imageBlob

blob =  _blobImage(image=cv2.cvtColor(cv2.imread("face.jpg"), cv2.COLOR_BGR2RGB), out_size = (300, 300), scaleFactor = 1.0, mean = (104.0, 177.0, 123.0))

# extract face
def extract_face(image, thresh_scale = (20, 20)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detect = mtcnn.MTCNN()
    bbox = detect.detect_faces(image)[0]['box']
    if bbox[2] > 20 or bbox[3] > 20: 
        img = image[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
    return img
img = extract_face(cv2.imread('face.jpg'))
cv2.imshow("a", img)
cv2.waitKey()

