from text_to_speech import *
from speech_to_text import *
from similary import *
import datetime
from test_api import *

def insert_inf(id, name):
    with open('database.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'name'])
        csvwriter.writerows([[id, name]])

arr1 = ["một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
arr2 = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

day_vietnamese_dict = {"Monday":"Thứ hai", "Tuesday":"Thứ ba", "Wednesday":"Thứ tư", "Thusday": "Thứ năm",
                        "friday":"Thứ sáu", "Saturday": "Thứ bảy", "Sunday":"Chủ nhật"}
fruit = {"apple" :5000, "banana":6000, "orange":7000}
vietnamese_name = ["cam", "táo", "chuối"]
# s = "bán cho tôi quả chuối"
def get_name(s):
    for i in vietnamese_name:
        if i in s:
            label = i
    return label

def get_num(str):
    token = str.split(" ")
    for i in token:
        if(i in arr1):
            numb = number[i]
        elif(i in arr2):
            numb = i 
    return numb


number ={"một": '1', "hai":'2', "ba":'3', "bốn": '4', "năm": '5', "sáu":'6', "bảy": '7', "tám": '8', "chín": '9'}

a = ''
x = ""
# print(int(a)*3)
count = 0
deal_num = 0
orange = 0
apple = 0
banana = 0
while(x != "tạm biệt"):
    a = speech_to_text()
    x = similary(a)
    # print(x)
    if x == "xin chào\n":
        text_to_speech("chào bạn tôi có thể giúp gì cho bạn")

    if x == "bây giờ là mấy giờ\n":
        time = datetime.datetime.now()
        time = time.strftime("%H:%M:%S").split(":")
        h = time[0]
        m = time[1]
        text_to_speech("bây giờ là {} giờ {} phút".format(h, m))
        
    if x == "hôm nay là thứ mấy\n":
        day = datetime.datetime.now()
        day = day.strftime("%A")
        day_vn = day_vietnamese_dict[day]
        text_to_speech("hôm nay là {}".format(day_vn))

    if x == "bán cho tôi quả cam\n":
        text_to_speech("bạn muốn mua bao nhiêu quả cam")
        num = speech_to_text()
        num1 = get_num(num)  
        count += fruit['orange']*int(num1)
        # print(count)
        orange += int(num1)
    if x == "bán cho tôi quả chuối\n":
        text_to_speech("bạn muốn mua bao nhiêu quả chuối")
        num = speech_to_text()
        num1 = get_num(num)  
        count += fruit['banana']*int(num1)
        banana += int(num1)
        # print(count)
    if x == "bán cho tôi quả táo\n":
        text_to_speech("bạn muốn mua bao nhiêu quả táo")
        num = speech_to_text()
        num1 = get_num(num)
        count += fruit['apple']*int(num1)
        apple += int(num1)
        # print(count)

    if x == "tính tiền\n":
        text_to_speech("của bạn hết {} đồng".format(count))
        print("tinh tien", count)
    if x =="tình hình dịch bệnh\n":
        num = get_covid()
        text_to_speech("Tổng số ca nhiễm hiện nay là {}".format(num))
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
        text_to_speech("bạn có muốn mua gì thêm không")
        out = str(input("yes/no:"))
        if (out == "yes"):
            continue
        else:
            text_to_speech("Mời bạn nhập thông tin")
            name = input("Mời bạn nhập tên: ")
            id = input("Mời bạn nhập mã thẻ: ")
            insert_inf(id, name)
            text_to_speech("đây là hóa đơn của bạn")
            print("khách hàng: ", name)
            print("mã thẻ: ", id )
            print("quả táo: ", apple)
            print("quả cam: ", orange)
            print("quả chuối: ", banana)
            print("tổng tiền: ", count)
