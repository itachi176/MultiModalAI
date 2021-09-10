import requests
from bs4 import BeautifulSoup

req = requests.get('https://ncov.moh.gov.vn/', verify=False)
soup = BeautifulSoup(req.text, "lxml")

# print(soup.find("div", {"id":"sailorTableArea"}))
arr = []
for i in soup.findAll("td", {"class":"text-danger-new"}):
    num = i.text
    if num == 0:
        num = 0 
    else:
        if "." in num:
            num = num.replace(".","")
        if '+' in num:
            num = num[1:]
    arr.append(int(num))
print("soca:", sum(arr))