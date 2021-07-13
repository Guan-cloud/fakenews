import pandas as pd

df = pd.read_csv('final_new.csv', dtype={'id': str})


df[['news']] = '"' + df[['news']] + '"'
# df[['id']] = df[['id']].apply(str)
df[['id']] = '"' + df[['id']] + '"'
df[['label']] = '"' + df[['label']] + '"'

print(df)
df.to_csv('final_guan.txt', index=False,header=None)

