from gtts import gTTS
import playsound
import urllib
from requests import get  # to make GET request
import requests
import json 

def text_to_speech(text):
    output = gTTS(text,lang="vi", slow=False)
    output.save("output.mp3")
    playsound.playsound('output.mp3', True)


# def text_to_speech(text):

#     url = 'https://api.fpt.ai/hmi/tts/v5'

#     payload = text
#     headers = {
#         'api-key': 'PD8zMEUUnOOx7kztRg5rsfjXpTnEweph',
#         'speed': '0',
#         'voice': 'leminh'
#     }

#     response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)

#     x = json.loads(response.text)
#     # return x['async']
#     download(x['async'], './a.mp3')
#     playsound.playsound("a.mp3", True)

# # a = test()
# # # print(type(a))

# def download(url, file_name):
#     # open in binary mode
#     with open(file_name, "wb") as file:
#         # get request
#         response = get(url)
#         # write to file
#         file.write(response.content)


