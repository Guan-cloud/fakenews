# -*- coding: utf-8 -*-
"""
File Name：     predict_utils
date：          2020/12/15
author:        'Hub'
"""
from transformers import InputExample, InputFeatures
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset
from collections import Counter
import csv
import numpy as np
import math


def load_data(filename):
    datas = pd.read_csv(filename).values.tolist()
    # print("datas:")
    # print(datas)
    return datas



def create_examples(filename):
    datas = pd.read_csv(filename,encoding='utf-8',dtype=str).values.tolist()
    # print(datas)
    examples = []
    for i, data in enumerate(datas):
        guid = data[0]
        # text_a = data[1].strip()
        # text_b = data[2].strip()

        # by hub just one sentence
        text_a = data[1].strip()
        examples.append(
            InputExample(
                guid=guid,
                # text_a=text_a,
                # text_b=text_b,

                # by hub just one sentence
                text_a=text_a,
                label=None
            )
        )
    return examples


def create_features(examples, tokenizer, max_len):
    features = []
    pad_on_left = False
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0
    mask_padding_with_zero = True
    for example in tqdm(examples, desc='convert examples to features'):
        # inputs = tokenizer.encode_plus(example.text_a, example.text_b,
        #                                add_special_tokens=True, max_length=max_len)
        # by hub just one sentence
        inputs = tokenizer.encode_plus(example.text_a, add_special_tokens=True,
                                       max_length=max_len, return_token_type_ids=True)
        # 就是 token 对应的句子id，值为0或1（0表示对应的token属于第一句，1表示属于第二句）。形状为(batch_size, sequence_length)。
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_len - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_len, "Error with input length {} vs {}".format(len(input_ids), max_len)
        assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(
            len(attention_mask), max_len
        )
        assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(
            len(token_type_ids), max_len
        )
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=None
            )
        )
    return features


def create_dataset(filename, tokenizer, max_len):
    examples = create_examples(filename)
    # print("1")
    # print(examples)
    features = create_features(examples, tokenizer, max_len)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_token_type_ids
    )
    ids = [example.guid for example in examples]
    return dataset, ids


def mean_re(logits):
    if len(logits) == 1:
        return logits[0]
    res = None
    for logit in logits:
        if res is None:
            res = np.array(logit)
        else:
            res += np.array(logit)
    res = res * (1 / len(logits))
    return res.tolist()


def mean(logits):
    if len(logits) == 1:
        return logits[0]
    res = None
    for logit in logits:
        if res is None:
            res = logit
        else:
            res += logit
    res = res / len(logits)
    return res


def vote_re(predictions):
    '''
    投票融合方法
    :param predictions:
    :return:
    '''
    if len(predictions) == 1:  # 没有多个预测结果就直接返回第一个结果
        return predictions[0]
    result = []
    num = len(predictions[0])
    for i in range(num):
        temp = []
        temp_floor = []
        for pred in predictions:
            temp.append(pred[i])

        for t in temp:
            temp_floor.append(math.floor(t))

        counter = Counter(temp_floor)
        re_most = counter.most_common()[0][0]
        re_sum = 0
        re_num = 0
        for t in temp:
            if re_most <= t <= re_most + 1:
                re_sum += t
                re_num += 1
        re = re_sum / re_num
        result.append(re)
    return result


def vote(predictions):
    '''
    投票融合方法
    :param predictions:
    :return:
    '''
    if len(predictions) == 1:  # 没有多个预测结果就直接返回第一个结果
        return predictions[0]
    result = []
    num = len(predictions[0])
    for i in range(num):
        temp = []
        for pred in predictions:
            temp.append(pred[i])
        counter = Counter(temp)
        result.append(counter.most_common()[0][0])
    return result


def write_result(filename, ids, predictions, is_int=True, retain_decimal=2):
    with open(filename, 'w', newline='', encoding='utf-8') as w:
        writer = csv.writer(w)
        writer.writerow(['id', 'label'])
        if is_int:
            for id, pred in zip(ids, predictions):
                writer.writerow([id, int(pred)])
        else:
            for id, pred in zip(ids, predictions):
                writer.writerow([id, round(pred, retain_decimal)])


if __name__ == '__main__':
    pre_1 = [1.23, 2.20, 3.33, 5.00, 0.0]
    pre_2 = [2.23, 3.20, 0.0, 4.54, 0.0]
    pre_3 = [1.70, 2.20, 3.33, 2.54, 0.0]
    pre_4 = [1.50, 2.70, 3.33, 3.54, 0.0]
    pre_5 = [0.10, 2.90, 3.33, 1.54, 0.0]
    vote_re([pre_1, pre_2, pre_3, pre_4, pre_5])
