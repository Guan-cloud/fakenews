import pandas as pd
import csv
df=pd.read_csv("final.csv",encoding='utf-8')
print(df)
col_name=df.columns.tolist()
print(col_name)
col_name.insert(0,'news')
df=df.reindex(columns=col_name)
print(col_name)
print(df)
df['news']='fakenews'
print(df)
data_new=df.to_csv("final_new.csv",index=False)
import pandas as pd
import os
# 加双引号
# data = pd.read_csv('final.csv', encoding='utf-8')
# data.to_csv('f2.csv',quoting=1,mode='w',index=False,encoding='utf-8')

# df=pd.read_csv('f.csv')
# print(df)
# df_new=df.to_csv('fake5.txt',index=False,header=None)
# print(df_new)

# data = pd.read_csv('f2.csv', encoding='utf-8')
# with open('f3.txt', 'a+', encoding='utf-8') as f:
#     f.write(('news' + '\t' + 'id' +'\t'+'label'+'\n'))
#     for line in data.values:
#         f.write((str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n'))
