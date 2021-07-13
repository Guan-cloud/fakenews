# import pandas as pd
# data1=pd.read_csv('f--left.csv')
# data2=pd.read_csv('f--right.csv')
# res=pd.merge(data1,data2,left_on=None, left_index=True, right_index=True, how='outer')
# res.to_csv('hello.csv',index=False)
# 导入模型和分词器
from transformers import BertTokenizer,BertModel,RobertaTokenizer, RobertaModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # Bert的分词器
bertmodel = BertModel.from_pretrained('bert-base-cased',from_tf=True).cuda() # load the TF model for Pytorch
text = " I love python ! "
# 对于一个句子，首尾分别加[CLS]和[SEP]。
text = "[CLS] " + text + " [SEP]"
# 然后进行分词
tokenized_text1 = tokenizer.tokenize(text)
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
# 分词结束后获取BERT模型需要的tensor
segments_ids1 = [1] * len(tokenized_text1)
tokens_tensor1 = torch.tensor([indexed_tokens1]).cuda() # 将list转为tensor
segments_tensors1 = torch.tensor([segments_ids1]).cuda()
# 获取所有词向量的embedding
word_vectors1 = bertmodel(tokens_tensor1, segments_tensors1)[0]
# 获取句子的embedding
sentenc_vector1 = bertmodel(tokens_tensor1, segments_tensors1)[1]

########################################
# 两个句子
########################################
# 在第二个句子末尾加个[EOS]
text2 = " I love Java! "
text3 = " I love C#! "
text = "[CLS] " + text2 + " [SEP]"
tokenized_text2 = tokenizer.tokenize(text)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
segments_ids2 = [0] * len(tokenized_text2) # 第一个句子的索引为0
text = text3 + " [EOS]"
tokenized_text3 = tokenizer.tokenize(text)
indexed_tokens3 = tokenizer.convert_tokens_to_ids(tokenized_text3)
segments_ids3 = [1] * len(tokenized_text3) # 第二个句子的索引为1
segments_ids = segments_ids2 + segments_ids3
indexed_tokens = indexed_tokens2  + indexed_tokens3
tokens_tensor = torch.tensor([indexed_tokens]).cuda() # 将list转为tensor
segments_tensors = torch.tensor([segments_ids]).cuda()
# 获取所有词向量的embedding
word_vectors = bertmodel(tokens_tensor, segments_tensors)[0]
# 获取句子的embedding
sentenc_vector = bertmodel(tokens_tensor, segments_tensors)[1]
