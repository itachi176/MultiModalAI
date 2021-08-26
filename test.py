import pytesseract as  tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2 
import re
from PIL import Image   
import time
a = time.time()
img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cong = r'--oem 3 --psm 6 outputbase digits'

x = tess.image_to_string(img, lang = "vie")
corr = []
name = []
token = x.split("\n")
token = token[:-1]
print(token)
# print(token[1])
for i in token:
    regex = re.search("^\d", i)
    if regex != None:
        # i = i.replace(" ", '')
        name.append(i)
#split data 
name1 = []
name2 = []
# print(name)
#get name 
for i in name:
    name1.append(i.split("SL")[0])
    name2.append(i.split("SL")[1])

for i in range(len(name1)):
    name1[i] = name1[i][:-2]

tenthuoc=[]
soluong=[]

for i in name1:
    tenthuoc.append(i.split(".")[1])
# for i in name1:
#     i=i[2:]
    # print(i)
print("ten thuoc: ", tenthuoc)

#lay so luong thuoc 
for i in range (len(name2)):
    name2[i] = name2[i][1:]
# print(name1)
print(time.time() - a)
# print(x)