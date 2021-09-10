# from os import nice
import cv2 
import pytesseract as  tess
from text_to_speech import *
from speech_to_text import *
from similary import *
import requests
import cv2
import numpy as np
import imutils
import re 
import time

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
corr_dict ={
    "fenoprofen":[[300, 50, 220],[250,50,220]],
    "paracetamol":[[300, 50, 220],[250,50,220]],
    "methorphan":[[300, -70, 220], [250, -70, 220]],
    "prospan": [[300, -70, 220], [250, -70, 220]],
    "berberin":[[250, -70, 110], [300, -70 ,110]],
    "tiffy": [[250, 50, 110],[300, 50, 110]],
    "decolgen": [[250, 50, 110],[300, 50, 110]],
    "vitamin C": [[250, 50, 55]],
    "seduxen":[[250, -70, 55]]
}

type_dict = {
    "thuốc đau đầu":["Fenoprofen", "paracetamol"],
    "thuốc đau bụng":["Berberin"],
    "thuốc ho": ["Methorphan", "Prospan"]
}

def scan():
    url = "http://192.168.1.4:8080/shot.jpg"
  
# While loop to continuously fetching data from the Url
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)
        cv2.imshow("Android_cam", img)
        cv2.imwrite("test.jpg", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(pytesseract.image_to_string(img))
        
        # cong = r'--oem 3 --psm 6 outputbase digits'
        start1 = time.time()
        x = tess.image_to_string(img, lang ='vie')
        y = x.split("\n")
        time_run1 = time.time()-start1
        # Press Esc key to exit
        if cv2.waitKey(1) == 27:
            return y, time_run1
            # break
    
    cv2.destroyAllWindows()


    # img = cv2.imread("./test.jpg")

    # # img = cv2.resize(img, (640, 360))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # print(pytesseract.image_to_string(img))
        
    # cong = r'--oem 3 --psm 6 outputbase digits'
    # x = pytesseract.image_to_string(img, config = cong, lang = 'vie')
    # y = x.split("\n")
    # # print(y)
    # # print(corr_dict[y[1]])
    # cv2.imshow("aa", img)
    # cv2.waitKey()
    return y


# text_to_speech("chào mừng bạn đến với cửa hàng chúng tôi")
# text_to_speech("mời bạn chọn 1, 2 hoặc 3")
# text_to_speech("1, nhập đơn thuốc thừ bàn phím")
# text_to_speech("2, đưa đơn thuốc lên camera để scan")
# text_to_speech("3, mời nói tên loại thuốc muốn mua")
print("1-----nhập đơn thuốc từ bàn phím")
print("2----- đưa đơn thuốc lên cam để scan")
print("3--------mời nói tên loại thuốc muốn mua")
a = int(input("mời bạn chọn:"))

arr = []
if a == 1:
    num = int(input("nhập số loại thuốc :"))
    for i in range (num):
        temp = input("nhập loại thuốc {} :".format(i+1))
        arr.append(temp)

    print("tọa độ của 2 loại thuốc là: {}, {}".format(corr_dict[arr[0]], corr_dict[arr[1]]))
    text_to_speech("Bạn mua thành công {} loại thuốc là {} và {}, xin chờ robot lấy thuốc và thanh toán".format(num, arr[0].lower(), arr[1]))

if a == 2:
    while(True):
        corr = []
        name = []
        token, time_run1 = scan()
        start2 = time.time()
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
            try:
                name1.append(i.split("SL")[0])
                name2.append(i.split("SL")[1])
            except:
                pass

        for i in range(len(name1)):
            name1[i] = name1[i][:-2]

        tenthuoc=[]
        soluong=[]
        
        print(name1)
        for i in name1:
            try:
                tenthuoc.append(i.split(".")[1])
            except:
                pass
        # for i in name1:
        #     i=i[2:]
            # print(i)
        print("aa", tenthuoc)

        #lay so luong thuoc 
        for i in range (len(name2)):
            name2[i] = name2[i][1:]
        print(name1)
        time_run2 = time.time() - start2
        print(time_run1+time_run2)
        conf = len(tenthuoc)/5
        with open('data_ocr.txt', 'a') as file:
            file.write(str(conf) +" "+ str(time_run2 + time_run1) + "\n")

    # text_to_speech("bạn mua thành công các loại thuốc sau:")
    # for i in range (len(name1)):
    #     text_to_speech("{} {}".format(name2[i], name1[i]))
    # for i in token:
    #     token_i = i.split(":")
    #     name.append(token_i[0])
    #     # print(corr_dict[token_i[0]])
    #     # print(token_i[1].strip())
    #     for j in range(int(token_i[1].strip())):
    #         corr.append(corr_dict[token_i[0]][j])
    # text_to_speech("bạn đã mua các loại thuốc sau")
    # for i, j in enumerate(name):
    #     text_to_speech(str(i+1) + j)
    # text_to_speech("mời chờ robot và thanh toán")

    print(corr)
    # print(token[0])
    # print("toạ độ của 2 loại thuốc là {}, {}".format(corr[0], corr[1]))
    # text_to_speech("bạn mua thành công {} loại thuốc là {} và {}, xin chờ robot lấy thuốc và thanh toán".format(len(corr), token[0].lower(), token[1].lower()))

if a == 3:
    text_to_speech("mời bạn đọc loại thuốc muốn mua")
    
    while(True):
        a = speech_to_text()
        a = similary(a)
        if a == "thuốc đau đầu\n":
            name = type_dict[a[:-1]]
            text_to_speech("chúng tôi đang có các loại thuốc sau")
            for i, j in enumerate(name):
                text_to_speech(str(i + 1) + j)
                print(str(i+1) + "------" + j)

            text_to_speech("mời bạn chọn loại thuốc")
            choose = int(input("mời bạn nhập số để chọn: "))
            text_to_speech("bạn đã chọn {}". format(name[choose-1]))
        

        if a == "thuốc ho\n":
            name = type_dict[a[:-1]]
            text_to_speech("chúng tôi đang có các loại thuốc sau")
            for i, j in enumerate(name):
                text_to_speech(str(i + 1) + j)
                print(str(i+1) + "------" + j)

            text_to_speech("mời bạn chọn loại thuốc")
            choose = int(input("mời bạn nhập số để chọn: "))
            text_to_speech("bạn đã chọn {}". format(name[choose-1]))
        text_to_speech("bạn muốn mua gì nữa không?")
        exit = input("yes/no:")
        if exit == "yes":
            pass 
        else:
            text_to_speech("mời đến quầy thanh toán và nhận thuốc")
            break
