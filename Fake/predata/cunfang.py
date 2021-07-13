import pandas as pd
# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
# pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 10240)#加了这一行那表格的一行就不会分段出现了
# pd.set_option('display.max_colwidth', 1000)
# pd.set_option('display.height', 1000)
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',500)

# train=pd.read_csv("../data/Constraint/handi/constraint_Hindi_Train - Sheet1.csv")
valid=pd.read_csv("../data/Constraint/handi/constraint_Hindi_Valid - Sheet1.csv")
valid.columns = ['id','post','label']
print(valid.count())
# print(train.id.value_counts())
# print(train.post.value_counts())
print(valid.label.value_counts())
# print(train['label'])

valid['defamation'] = 0
valid['fake'] = 0
valid['hate'] = 0
valid['offensive'] = 0
valid['non-hostile'] = 0


for i,row in valid.iterrows():
    if(row['label'] == "non-hostile"):#435
        valid.loc[i,'non-hostile'] = 1
    if (row['label'] == "fake"):#144
        valid.loc[i, 'fake'] = 1
    if (row['label'] == "hate"):#68
        valid.loc[i, 'hate'] = 1
    if (row['label'] == "offensive"):#57
        valid.loc[i, 'offensive'] = 1
    if (row['label'] == "defamation"):#43
        valid.loc[i, 'defamation'] = 1
    if (row['label'] == "hate,offensive"):#23
        valid.loc[i, 'hate'] = 1
        valid.loc[i, 'offensive'] = 1
    if (row['label'] == "defamation,offensive"):#11
        valid.loc[i, 'defamation'] = 1
        valid.loc[i, 'offensive'] = 1
    if (row['label'] == "defamation,hate"):#10
        valid.loc[i, 'defamation'] = 1
        valid.loc[i, 'hate'] = 1
    if (row['label'] == "fake,offensive"):#4
        valid.loc[i, 'fake'] = 1
        valid.loc[i, 'offensive'] = 1
    if (row['label'] == "defamation,hate,offensive"):#4
        valid.loc[i, 'defamation'] = 1
        valid.loc[i, 'hate'] = 1
        valid.loc[i, 'offensive'] = 1
    if (row['label'] == "defamation,fake"):#4
        valid.loc[i, 'defamation'] = 1
        valid.loc[i, 'fake'] = 1
    if (row['label'] == "fake,hate"):#3
        valid.loc[i, 'fake'] = 1
        valid.loc[i, 'hate'] = 1
    if (row['label'] == "defamation,fake,offensive"):#3
        valid.loc[i, 'defamation'] = 1
        valid.loc[i, 'fake'] = 1
        valid.loc[i, 'offensive'] = 1
    if (row['label'] == "defamation,fake,hate"):#1
        valid.loc[i, 'defamation'] = 1
        valid.loc[i, 'fake'] = 1
        valid.loc[i, 'hate'] = 1
    if (row['label'] == "defamation,fake,hate,offensive"):#1
        valid.loc[i, 'defamation'] = 1
        valid.loc[i, 'fake'] = 1
        valid.loc[i, 'hate'] = 1
        valid.loc[i, 'offensive'] = 1




#以下保存指定的列到新的csv文件，index=0表示不为每一行自动编号，header=1表示行首有字段名称
valid.to_csv('handi_valid_preprocessing.csv',columns=['id','post','label','defamation','fake','hate','offensive','non-hostile'],index=0,header=1)