import pandas as pd

# test_news = pd.read_csv('result_a.csv')
#
# test_news = test_news[['tweet_id', 'task1']]
# test_news_pre = pd.read_csv('english_test_1509.csv')
# test_news['ID'] = test_news_pre['ID']
#
# test_news.loc[test_news['task1'] ==0,'task1']= 'NOT'
# test_news.loc[test_news['task1'] ==1,'task1']= 'HOF'
#
#
# test_news.to_csv('test_a.csv', index=0)

# test_news = pd.read_csv('result_b.csv')
#
# test_news = test_news[['tweet_id', 'task2']]
# test_news_pre = pd.read_csv('english_test_1509.csv')
# test_news['ID'] = test_news_pre['ID']
#
# test_news.loc[test_news['task2'] ==0,'task2']= 'NONE'
# test_news.loc[test_news['task2'] ==1,'task2']= 'PRFN'
# test_news.loc[test_news['task2'] ==2,'task2']= 'OFFN'
# test_news.loc[test_news['task2'] ==3,'task2']= 'HATE'
#
#
#
# test_news.to_csv('test_b.csv', index=0)
import pandas as pd

pd1 = pd.read_csv('../prediction_result/1a.csv')
pd2 = pd.read_csv('../prediction_result/1b.csv')
pd3 = pd.read_csv('../prediction_result/1croberta.csv')
pd4 = pd.read_csv('../prediction_result/2aclassfication.csv')

print(pd1)
df = pd.DataFrame({'id':pd1['id'], 'is_humor': pd1['label'], 'humor_rating':pd2['label'], 'humor_controversy':pd3['label'], 'offense_rating':pd4['label']})

df['id'] = df['id'].astype(int)

df['is_humor'] = df['is_humor'].astype(int)
df.to_csv('../prediction_result/result.csv', index=False)


# pd1 = pd.read_csv('result_tamil.csv')
# pd2 = pd.read_csv('test_tamil.csv')
#
# # df = pd.DataFrame({'id':pd1['id'],'text':pd2['text'],'label':pd1['label']})
# df = pd.DataFrame({'id':pd1['id'],'text':pd2['text']})
# df.to_csv('final_tamil_result.csv', index=False)



#
# trail.loc[trail['label'] =='0','label']= 'Non_hope_speech'
# trail.loc[trail['label'] =='1','label']= 'Hope_speech'
# trail.loc[trail['label'] =='2','label']= 'not-English'


# trail = pd.read_csv('result_eng.tsv')
# trail['label'] = trail['label'].replace(1, 'Non_hope_speech')
# trail['label'] = trail['label'].replace(1, 'Hope_speech')
# trail['label'] = trail['label'].replace(2, 'not-English')
# trail.to_csv('result.tsv', index=False)


# trail['label'] = trail['label'].str.replace(0, 'Non_hope_speech')
# trail['label'] = trail['label'].str.replace(1, 'Hope_speech')
# df = df['label'].str.replace(2, 'not-English')


