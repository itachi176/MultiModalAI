import requests
import json
url = "https://api.apify.com/v2/key-value-stores/ZsOpZgeg7dFS1rgfM/records/LATEST?fbclid=IwAR1UCKt-lM0mITqxyalzx-XdQ3cFYX51Il_7kU0X79sS5LDZwdIp7FFPAxg&utm_source=j2team&utm_medium=url_shortener"
def get_covid():
    r = requests.get(url)
    x = json.loads(r.text)
    print(x['infected'])
    return x['infected']