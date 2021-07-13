import pandas as pd
df=pd.read_excel('test.xlsx')
print(df)
df.to_csv('test.csv',index=False,encoding='utf-8')