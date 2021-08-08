import requests

url = 'https://api.fpt.ai/hmi/tts/v5'

payload = 'xin chào'
headers = {
    'api-key': 'PD8zMEUUnOOx7kztRg5rsfjXpTnEweph',
    'speed': '',
    'voice': 'banmai'
}

response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)

print(response.text)