from speech_to_text import speech_to_text
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
    a = ''
    while(a != "tạm biệt"):
        a = speech_to_text()
        if a == "Xin chào":
            text_to_speech("chào hoàng, tôi có thể giúp gì cho bạn")
        if a == "tạm biệt":
            text_to_speech("Hẹn gặp lại hoàng")

th1 = Thread(target=video)
th1.start()
th2 = Thread(target=speak)
th2.start()