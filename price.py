import pandas as pd 

df = pd.read_csv("./data.csv")
def price_object(name):
    price = df[df['Name']==name]['price'].values[0]
    return price

# print(price_object('cell phone'))
