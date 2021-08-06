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


def test():

    url = 'https://api.fpt.ai/hmi/tts/v5'

    payload = 'chào hoàng tôi có thể giúp gì cho bạn '
    headers = {
        'api-key': 'PD8zMEUUnOOx7kztRg5rsfjXpTnEweph',
        'speed': '',
        'voice': 'banmai'
    }

    response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)

    x = json.loads(response.text)
    return x['async']

a = test()
# print(type(a))

def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

download("https://file01.fpt.ai/text2speech-v5/short/2021-08-06/banmai.0.9f745608c0120df9e68b7c354ccfddb4.mp3", "a.mp3")
playsound.playsound("a.mp3", True)
