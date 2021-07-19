import pandas as pd
import re
import numpy as np

buy = pd.read_csv('buy.csv', encoding = 'ISO-8859-1')
abt = pd.read_csv('abt.csv',encoding = 'ISO-8859-1')


buy['manufacturer'] = buy['manufacturer'].fillna(buy['name'])
buy['manufacturer'] = buy['manufacturer'].str.split().str.get(0).str.lower()

manufacturers = buy['manufacturer'].unique().tolist()
blocks = buy.groupby(by=['manufacturer'])

abtBlock = []
buyBlock = []
def findManf(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search
for abt_index, abt_rec in abt.iterrows():
    for manufacturer in manufacturers:
        if(findManf(manufacturer)(abt_rec['name'])):
            abtBlock.append((manufacturer,abt_rec['idABT']))
            break
    else:
        abtBlock.append((np.nan,abt_rec['idABT']))
        

for manufacturer in manufacturers:
    buy_block = blocks.get_group(manufacturer)
    for buy_index, buy_rec in buy_block.iterrows():      
        buyBlock.append((manufacturer,buy_rec['idBuy']))
    
        
abt_blocks = pd.DataFrame(abtBlock, columns=['block_key', 'product_id'])
buy_blocks = pd.DataFrame(buyBlock, columns=['block_key', 'product_id'])

abt_blocks.to_csv('abt_blocks.csv', index=False)
buy_blocks.to_csv('buy_blocks.csv', index=False)