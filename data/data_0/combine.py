import pandas as pd
# df=pd.read_csv('train.csv')
# df["Text"]=df["Topic"]+' '+df["Source"]+' '+df['Text']
# df.to_csv('train-new.csv',columns=['Id','Category','Text'],index=False)


df_dev=pd.read_csv('evaluate.csv',encoding='utf-8')
df_dev["Text"]=df_dev["Topic"]+' '+df_dev["Source"]+' '+df_dev['Text']
df_dev.to_csv('dev-new.csv',columns=['Id','Category','Text'],index=False)

