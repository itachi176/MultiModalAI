import torch 
import  cv2
import os 
import time
# from speech.text_to_speech import *
from gtts import gTTS
from threading import Thread 
from speech.similary import *
from speech.speech_duration import mutagen_length

while(True):
    x = speech_to_text()
    output = gTTS(x,lang="vi", slow=False)
    output.save("./data/output.mp3")
    time = mutagen_length('./data/output.mp3')
    with open('time1.txt', 'a') as f:
        f.writelines(str(time)+'\n')
    try:
        x= similary(x)
        print(x)
    except:
        pass
    
    if x == "lấy cái hộp\n":
        print('red')
    if x =="lấy cái vòng tròn\n":
        print("green")
    if x == "lấy cục pin\n":
        print("yellow")
    