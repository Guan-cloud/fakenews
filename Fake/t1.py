
import pandas as pd

# df1 = pd.read_csv(r'f--left.csv', encoding='utf-8')#读取第一个文件
#
# df2 = pd.read_csv(r'f--right.csv', encoding='utf-8')#读取第二个文件
#
# outfile = pd.merge(df1, df2,  left_index=True, right_index=True)#文件合并 left_on左侧DataFrame中的列或索引级别用作键。right_on 右侧


# outfile.to_csv(r'a.csv', index=False,encoding='utf-8')#输出文件
# pd.read_csv('a.csv',usecols=[1,2,3]).to_csv('b.csv',index=False,header=False)



# data=pd.read_csv('f.csv',encoding='utf-8')
# with open('h.csv','w',encoding='utf-8') as f:
#     for line in data.values:
#         f.write((str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n'))

# import pandas as pd
# import numpy as np
# df=pd.read_csv('f--right.csv')
# col_name=df.columns.tolist()
# print(col_name)
# col_name.insert(0,'name')
# df=df.reindex(columns=col_name)
# print(col_name)
# df['name'] = 'fakenews'
# print(df)
# df.to_csv("f.csv",index=False)

import pandas as pd
data=pd.read_csv('f.csv',encoding='utf-8')
data.to_excel('h.xlsx',encoding='utf-8',index=False,header=False)




