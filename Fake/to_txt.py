import pandas as pd
import os

data = pd.read_csv('f.csv', encoding='utf-8')

# data.loc[data['label'] == 0 ,'label']= 'True'
# data.loc[data['label'] == 1 ,'label']= 'Fake'


with open('Ree.txt', 'a+', encoding='utf-8') as f:
    # f.write(('id' + ',' + 'label' + '\n'))
    for line in data.values:
        f.write((str(line[0])+','+str(line[1])+'\n'))