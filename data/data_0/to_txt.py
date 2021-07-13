import pandas as pd
import os

data = pd.read_csv('f2.csv', encoding='utf-8',dtype=str,header=None)
print(data)
# data.loc[data['label'] == 0 ,'label']= 'True'
# data.loc[data['label'] == 1 ,'label']= 'Fake'


with open('guan.txt', 'a+', encoding='utf-8') as f:
    # f.write(('id' + ',' + 'label' + '\n'))
    for line in data.values:
        # print(str(line[1]))
        f.write((str(line[0])+'\t'+str(line[1])+'\n'))