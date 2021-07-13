
# f1=open(filename,'rb')
# data=f1.readlines()
# for line in data:
#     print(line)
# import pandas as pd
# pf=pd.read_csv('test.csv')
# pf['Text'].replace('\\n',' ',regex=True)
# pf.to_csv('final.csv',index=False)
with open('test.csv', 'r',encoding='utf-8') as txtReader:
    with open('new_test.csv', 'w',encoding='utf-8') as txtWriter:
        for line in txtReader.readlines():

            if line.strip():

                line = line.replace('\\n', '')
                txtWriter.write(line)