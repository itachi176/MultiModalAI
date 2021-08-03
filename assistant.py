from text_to_speech import *
from speech_to_text import *
from similary import *
import random 

fruit = {"apple" :5000, "banana":6000, "orange":7000}
vietnamese_name = ["cam", "táo", "chuối"]
# s = "bán cho tôi quả chuối"
def get_name(s):
    for i in vietnamese_name:
        if i in s:
            label = i
    return label

a = ''
x = ""
# print(int(a)*3)
count = 0
deal_num = 0
while(x != "tạm biệt\n"):
    a = speech_to_text()
    x = similary(a)
    if x == "xin chào\n":
        text_to_speech("chào bạn tôi có thể giúp gì cho bạn")
    if x == "bán cho tôi quả cam\n":
        text_to_speech("bạn muốn mua bao nhiêu quả cam")
        num = speech_to_text()
        count += fruit['orange']*int(num)
        print(count)
    if x == "bán cho tôi quả chuối\n":
        text_to_speech("bạn muốn mua bao nhiêu quả chuối")
        num = speech_to_text()
        count += fruit['banana']*int(num)
        print(count)
    if x == "bán cho tôi quả táo\n":
        text_to_speech("bạn muốn mua bao nhiêu quả táo")
        num = speech_to_text()
        count += fruit['apple']*int(num)
        print(count)

    if x == "tính tiền\n":
        text_to_speech("của bạn hết {} đồng".format(count))

    if x == "giảm giá cho tôi\n":
        if  deal_num < 2:
            text_to_speech("{} đồng nhé".format(count - 5000))
            count = count - 5000
            deal_num += 1 
        elif deal_num >= 2:
            text_to_speech("xin lỗi bạn, không giảm nữa")
    if x == "tạm biệt\n":
        text_to_speech("hẹn gặp lại {}".format(name))
        
    if x == "ok\n":
        text_to_speech("Mời bạn nhập thông tin")
        name = input("Mời bạn nhập tên: ")
        id = input("Mời bạn nhập mã thẻ: ")
        text_to_speech("đây là hóa đơn của bạn")
        print("khách hàng: ", name)
        print("mã thẻ: ", id )
        print("tổng tiền: ", count)