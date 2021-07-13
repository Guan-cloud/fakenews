import pandas as pd
pf=pd.read_csv('xlm-4.csv')
list=pf['label'].values.tolist()
print(list)