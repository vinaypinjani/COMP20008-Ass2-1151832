import pandas as pd
from textdistance import cosine
import re


abt = pd.read_csv('abt_small.csv', encoding = 'ISO-8859-1')
buy = pd.read_csv('buy_small.csv', encoding='ISO-8859-1')


match_id = []
thresh = 0.75
for abt_index, abt_rec in abt.iterrows():
    max_sim=0
    abt_manufacturer = abt_rec['name'].split()[0]
    abt_serial_id = abt_rec['name'].split()[-1]
    abt_name = abt_rec['name']
    abt_id = abt_rec['idABT']
    for buy_index, buy_rec in buy.iterrows():
        buy_name = buy_rec['name']
        buy_id = buy_rec['idBuy']
        buy_manufacturer = buy_rec['manufacturer']
        if re.compile((re.sub(r'[^\w\s]','',abt_serial_id)), flags=re.IGNORECASE).search(re.sub(r'[^\w\s]','',buy_name)):            
            match_id.append((abt_id, buy_id))
            buy.drop(buy_index, inplace=True)
            max_sim=0
            break
        if re.compile((re.sub(r'[^\w\s]','',abt_manufacturer)), flags=re.IGNORECASE).search(re.sub(r'[^\w\s]','',buy_manufacturer)):
            words = [(re.sub(r'[^\w\s]','',word)) for word in buy_name.split()]
            sim = max([cosine.normalized_similarity(word, abt_serial_id) for word in words])
            if(sim > max_sim):
                max_sim = sim
                max_id = buy_id


    if (max_sim > thresh):
        match_id.append((abt_id, max_id))




linked = pd.DataFrame(match_id, columns=['idAbt', 'idBuy'])
linked.to_csv('task1a.csv', index=False)






