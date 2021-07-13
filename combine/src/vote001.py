##########hard vote to result 1-4

import pandas as pd
import numpy as np
import os
from pprint import pprint
DATA_DIR = '../data/'
print(DATA_DIR)
files = os.listdir(DATA_DIR)
files = [i for i in files if i[0]!='.']
print("len(files):",len(files))
print("files:",files)

sub_exp_df = pd.read_csv('../submit_example.csv')
print("sub_exp_df",sub_exp_df)
df_merged = sub_exp_df.drop(['label'], axis=1)
print("df_merged:",df_merged)
# df_merged = pd.read_csv(DATA_DIR + files[0])
for file in files:
    tmp_df = pd.read_csv(DATA_DIR + file)
    df_merged = df_merged.merge(tmp_df, how='left', on='id')
    print(df_merged.shape)
pprint(df_merged.head(10))

def work(pres):
    count = [0,0,0]
    for i in pres:
        count[i] += 1
    out = count.index(max(count))
    return out

tmp_arr = np.array(df_merged.iloc[:,1:])
label_voted = [work(line) for line in tmp_arr]
print("label_voted:",label_voted)

df_summit = df_merged[['id']]
print("df_summit:",df_summit)

df_summit['label'] = label_voted

SUMMIT_DIR = 'E:/NLPProject/FakeNews/submit'
print("SUMMIT_DIR",SUMMIT_DIR)
df_summit[['id','label']].to_csv(SUMMIT_DIR + 'result_2.csv',index=False)
print(df_summit.shape)

###ceshi
def diff_df(fname1, fname2):
    sub_exp_df = pd.read_csv('../submit_example.csv')
    df_3 = sub_exp_df.drop(['label'], axis=1)

    df_1 = pd.read_csv(fname1)
    df_2 = pd.read_csv(fname2)

    df_3 = df_3.merge(df_1, how='left', on='id')
    df_3 = df_3.merge(df_2, how='left', on='id')
    print(df_3.shape)
    df_diff = df_3[df_3['label_x'] != df_3['label_y']]
    print(df_diff.shape)
    return df_diff


