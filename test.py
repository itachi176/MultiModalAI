from threading import Thread
import cv2
import time 
from text_to_speech import *

def video():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def speak():
    a = input("nhap gi do: ")
    if a == "xin chao":
        text_to_speech("chào hoàng")
th1 = Thread(target=video)
th1.start()
th2 = Thread(target=speak)
th2.start()