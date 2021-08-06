import requests

url = 'https://api.fpt.ai/hmi/asr/general'
payload = open('output.mp3', 'rb').read()
headers = {
    'api-key': 'PD8zMEUUnOOx7kztRg5rsfjXpTnEweph'
}

response = requests.post(url=url, data=payload, headers=headers)

print(response.json())