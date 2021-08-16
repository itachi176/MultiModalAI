import cv2 
import pytesseract
from text_to_speech import *
from speech_to_text import *
from similary import *
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
    img = cv2.imread("./a.jpg")
    # img = cv2.resize(img, (640, 360))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(pytesseract.image_to_string(img))
    cong = r'--oem 3 --psm 6 outputbase digits'
    x = pytesseract.image_to_string(img, config = cong, lang = 'vie')
    y = x.split("\n")
    # print(y)
    # print(corr_dict[y[1]])
    cv2.imshow("aa", img)
    cv2.waitKey()
    return y


# text_to_speech("chào mừng bạn đến với cửa hàng chúng tôi")
# text_to_speech("mời bạn chọn 1, 2 hoặc 3")
# text_to_speech("1, nhập đơn thuốc thừ bàn phím")
# text_to_speech("2, đưa đơn thuốc lên camera để scan")
# text_to_speech("3, mời nói tên loại thuốc muốn mua")
# print("1-----nhập đơn thuốc từ bàn phím")
# print("2----- đưa đơn thuốc lên cam để scan")
# print("3--------mời nói tên loại thuốc muốn mua")
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
    corr = []
    name = []
    token = scan()
    token = token[:-1]
    print(token)
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
