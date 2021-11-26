from gtts import gTTS
import playsound
import urllib
from requests import get  # to make GET request
import requests
import json 
import os 

def text_to_speech(text):
    output = gTTS(text,lang="vi", slow=False)
    output.save("../data/output.mp3")
    playsound.playsound('../data/output.mp3', True)

# def download(url, file_name):
#     if os.path.isfile(file_name):
#         os.remove(file_name)
#     # open in binary mode
#     with open(file_name, "wb") as file:
#         # get request
#         response = get(url)
#         # write to file
#         file.write(response.content)

# def text_to_speech(text):

#     url = 'https://api.fpt.ai/hmi/tts/v5'

#     payload = text
#     headers = {
#         'api-key': 'PD8zMEUUnOOx7kztRg5rsfjXpTnEweph',
#         'speed': '',
#         'voice': 'leminh'
#     }

#     response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)

#     x = json.loads(response.text)
#     # return x['async']
#     download(x['async'], './file.mp3')
#     playsound.playsound("file.mp3", True)

# # a = test()
# # print(type(a))
text_to_speech("hoang dep trai")


