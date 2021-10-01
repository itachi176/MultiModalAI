from speech.speech_to_text import *
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import copy


# a = speech_to_text()


def cosineSim(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range (len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
        
    return sumxy/math.sqrt(sumxx*sumyy)

def similary(a):
    TfidfVectorizer1 = TfidfVectorizer(ngram_range=(1,2))   

    tf_idf1 = TfidfVectorizer1.fit_transform([a, "xin chào"])
    sim = cosineSim(tf_idf1.toarray()[0], tf_idf1.toarray()[1])
    # print(sim)


    with open('./data/data.txt', 'r') as file:
        temp = 0
        str = ''
        for i in file:
            tf_idf1 = TfidfVectorizer1.fit_transform([a, i])
            sim = cosineSim(tf_idf1.toarray()[0], tf_idf1.toarray()[1])
            # print(sim)
            if(sim > temp):
                str = copy.copy(i)
                temp = sim
    # str = str.strip()
    return str

def similary_number(a):
    TfidfVectorizer1 = TfidfVectorizer(ngram_range=(1,2))   

    tf_idf1 = TfidfVectorizer1.fit_transform([a, "xin chào"])
    sim = cosineSim(tf_idf1.toarray()[0], tf_idf1.toarray()[1])
    # print(sim)
    arr = ["một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín","1", "2", "3", "4", "5", "6", "7", "8", "9"]

    temp = 0
    str = ''
    for i in arr:
        tf_idf1 = TfidfVectorizer1.fit_transform([a, i])
        sim = cosineSim(tf_idf1.toarray()[0], tf_idf1.toarray()[1])
        # print(sim)
        if(sim > temp):
            str = copy.copy(i)
            temp = sim
            
    str = str.strip()
    return str

# print(similary('xin chao'))