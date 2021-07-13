import pandas as pd
import numpy as np


train_df = pd.read_csv("evaluate.csv",encoding='utf-8')
# dev_df = pd.read_csv("dev-old.csv",encoding='utf-8')
# test_df=pd.read_csv('test-old.csv',encoding='utf-8')
train_df = train_df[['id', 'label',]]
# dev_df = dev_df[['Id', 'Category', 'Topic','Source','Headline','Text','Link']]
label_list = []
count = ["True", "Fake"]


def label_encode(label):
    label_encode_train = 'True'
    if label == 1:
        label_encode_train = 'Fake'
    return label_encode_train

train_df['label'] = train_df['label'].apply(lambda x :label_encode(x))
# dev_df['Category'] = dev_df['Category'].apply(lambda x :label_encode(x))
train_df.to_csv('evalute-new.csv', index=False)
# dev_df.to_csv('dev.csv', index=False)
# test_df.to_csv('test.csv',index=False)
# # label_list=list(set(label_list))
# label_list.sort()
# print(label_list)
# print(count)
