import cv2 
import pytesseract
from text_to_speech import *

corr_dict ={
    "thuốc đau đầu":[[300, 50, 220],[250,50,220]],
    "thuốc ho":[[300, -70, 220], [250, -70, 220]],
    "thuốc đau bụng":[[250, -70, 110], [300, -70 ,110]],
    "thuốc cảm cúm":[[250, 50, 110],[300, 50, 110]],
    "vitamin C": [[250, 50, 55]],
    "thuốc ngủ":[250, -70, 55]
}
def scan():
    img = cv2.imread("./ocr.png")
    img = cv2.resize(img, (640, 360))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(pytesseract.image_to_string(img))
    cong = r'--oem 3 --psm 6 outputbase digits'
    x = pytesseract.image_to_string(img, config = cong)
    y = x.split("\n")
    # print(y)
    # print(corr_dict[y[1]])
    cv2.imshow("aa", img)
    cv2.waitKey()
    return y

print(corr_dict["thuốc đau đầu"][0])
# text_to_speech("chào mừng bạn đến với cửa hàng chúng tôi")
# text_to_speech("mời bạn chọn 1 hoặc 2")
# text_to_speech("1, nhập đơn thuốc thừ bàn phím")
# text_to_speech("2, đưa đơn thuốc lên camera để scan")
# print("1-----nhập đơn thuốc từ bàn phím")
# print("2----- đưa đơn thuốc lên cam để scan")
# a = int(input("mời bạn chọn:"))

# arr = []
# if a == 1:
#     num = int(input("nhập số loại thuốc :"))
#     for i in range (num):
#         temp = input("nhập loại thuốc {} :".format(i+1))
#         arr.append(temp)

#     print("tọa độ của 2 loại thuốc là: {}, {}".format(corr_dict[arr[0]], corr_dict[arr[1]]))
#     text_to_speech("Bạn mua thành công {} loại thuốc là {} và {}, xin chờ robot lấy thuốc và thanh toán".format(num, arr[0].lower(), arr[1]))

# if a == 2:
#     corr = []
#     token = scan()
#     for i in token:
#         if i in corr_dict:
#             corr.append(corr_dict[i])
#     print("toạ độ của 2 loại thuốc là {}, {}".format(corr[0], corr[1]))
#     text_to_speech("bạn mua thành công {} loại thuốc là {} và {}, xin chờ robot lấy thuốc và thanh toán".format(len(corr), token[0].lower(), token[1].lower()))
