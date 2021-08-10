from text_to_speech import *
from speech_to_text import *
from similary import *
import datetime
from test_api import *
from lib import *
from threading import Thread
import csv

# data = my_data()
# # data = shuffle(data)
# train = data 
# test = data[:20]
# X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
# print(X_train.shape)
# y_train = [i[1] for i in train]
# X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
# print(X_test.shape)
# y_test = [i[1] for i in test]

# from tensorflow.python.framework import ops
# ops.reset_default_graph()
# convnet = input_data(shape=[50,50,1])
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# # 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 128, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)

# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
# convnet = fully_connected(convnet, 3, activation='softmax')
# convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
# model = tflearn.DNN(convnet, tensorboard_verbose=1)
# model.fit(X_train, y_train, n_epoch=10)



def insert_inf(id, name):
    with open('database.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'name'])
        csvwriter.writerows([[id, name]])


# s = "bán cho tôi quả chuối"


def speak():
    arr1 = ["một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    arr2 = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    day_vietnamese_dict = {"Monday":"Thứ hai", "Tuesday":"Thứ ba", "Wednesday":"Thứ tư", "Thusday": "Thứ năm",
                            "friday":"Thứ sáu", "Saturday": "Thứ bảy", "Sunday":"Chủ nhật"}
    fruit = {"apple" :5000, "banana":6000, "orange":7000}
    vietnamese_name = ["cam", "táo", "chuối"]
    number ={"một": '1', "hai":'2', "ba":'3', "bốn": '4', "năm": '5', "sáu":'6', "bảy": '7', "tám": '8', "chín": '9'}
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
            # day_vn = day_vietnamese_dict[day]
            text_to_speech("hôm nay là {}".format(day_vietnamese_dict[day]))
            # text_to_speech(day_vn)

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
                text_to_speech("đây là hóa đơn của bạn")
                # print("khách hàng: ", name)
                # print("mã thẻ: ", id )
                print("quả táo: ", apple)
                print("quả cam: ", orange)
                print("quả chuối: ", banana)
                print("tổng tiền: ", count)
def video():
    text_to_speech("Mời bạn nhập thông tin")
    name = input("Mời bạn nhập tên: ")
    id = input("Mời bạn nhập mã thẻ: ")
    insert_inf(id, name)
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    temp = 0
    while(True):
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        # centerH = frame.shape[0] // 2
        # centerW = frame.shape[1] // 2
        # sizeboxW = 300
        # sizeboxH = 400
        # cv2.rectangle(frame, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
        #               (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(frame, 1.3, 5)
        for x,y,w,h in faces:
            temp +=1
            cv2.rectangle(frame, (x, y), (x+w, y+h),(255,0,0), 1)
            cv2.imwrite("./dataset/"+ name +str(temp)+".jpg", frame_gray[y:y+h, x:x+w])
        cv2.imshow("frame", frame)
        if cv2.waitKey(100) & 0xFF==ord('q'):
            break
        if len(os.listdir("./dataset/")) > 100:
            # print("done")
            break


    cv2.destroyAllWindows()

th2 = Thread(target=speak)
th2.start()
th1 = Thread(target=video)
th1.start()

