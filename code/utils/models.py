from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.modeling_roberta import RobertaModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

from transformers.modeling_bert import BERT_INPUTS_DOCSTRING

from transformers.file_utils import add_start_docstrings_to_callable


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))



class BertCNNForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, need_cnn=False, dynamic_fusion=False):
        super(BertCNNForSequenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.is_dynamic_fusion = dynamic_fusion
        self.need_cnn = need_cnn
        self.init_weights()
        self.filter_sizes_ = (3, 4, 5)  # 卷积核尺寸
        self.num_filters_ = 256  # 卷积核数量(channels数)
        self.hidden_size_ = 768
        self.output_dim = config.hidden_size
        self.num_labels = config.num_labels
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters_, (k, self.hidden_size_)) for k in self.filter_sizes_])
        self.dropout = nn.Dropout(0.2)

        self.dropout_fusion = nn.ModuleList(
            [nn.Dropout(0.5) for _ in range(5)]
        )
        self.classifier = nn.Linear(self.output_dim, config.num_labels)
        self.wights = nn.Parameter(torch.rand(13, 1))

    def conv_and_pool(self, x, conv):
        # x = F.relu(conv(x)).squeeze(3)
        x = mish(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        hidden_states = outputs[2]
        if self.is_dynamic_fusion:
            for param in self.bert.parameters():
                param.requires_grad = True
            batch_size = sequence_output.shape[0]
            sq_len = sequence_output.shape[1]
            all_hidden_states = torch.cat(hidden_states).view(13, batch_size, sq_len, 768)
            attention_all_hidden_states = torch.sum(all_hidden_states * self.wights.view(13, 1, 1, 1), dim=[1, 2, 3])
            attention_all_hidden_states = F.softmax(attention_all_hidden_states.view(-1), dim=0)
            feature = torch.sum(all_hidden_states * attention_all_hidden_states.view(13, 1, 1, 1), dim=[0])
            sequence_output = feature
        if self.need_cnn:
            out = sequence_output.unsqueeze(1)
            out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
            sequence_output = self.dropout(out)
        elif self.is_dynamic_fusion:
            sequence_output = sequence_output[:, 0]
        else:
            sequence_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                predicts = logits.view(-1, self.num_labels)
                labels = labels.view(-1)
                ones = torch.sparse.torch.eye(self.num_labels).cuda()
                labels = ones.index_select(0, labels)
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(predicts, labels)
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs

class BertForSequenceClassificationlstm(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 考虑多分类的问题
        self.num_labels = config.num_labels
        # 调用bert预训练模型
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.lstm = []
        lstm_hidden_size = 64
        lstm_layers = 1
        for i in range(lstm_layers):
            self.lstm.append(nn.LSTM(config.hidden_size if i == 0 else lstm_hidden_size * 4, lstm_hidden_size,
                                     num_layers=1, bidirectional=True, batch_first=True).cuda())
        self.lstm = nn.ModuleList(self.lstm)
        # torch.Size([2, 256])
        # self.classifier = nn.Linear(args.lstm_hidden_size * 2, self.config.num_labels)
        # 在预训练的BERT上加上一个全连接层，用于微调分类模型
        self.classifier = nn.Linear(lstm_hidden_size * 2, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        last_hidden_state = outputs[0]
        # last_hidden_state.shape: [batch_size, sequence_length, hidden_size]

        for lstm in self.lstm:
            try:
                lstm.flatten_parameters()
            except:
                pass

            # print(last_hidden_state)
            output, (h_n, c_n) = lstm(last_hidden_state)
            # h_n.shape: [batch, num_layers*num_directions == 2, gru_hidden_size]    batch_size first

        x = h_n.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()
        # x.shape: [batch, 2 * gru_hidden_size]

        x = self.dropout(x)
        logits = self.classifier(x)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                predicts = logits.view(-1, self.num_labels)
                labels = labels.view(-1)

                ones = torch.sparse.torch.eye(self.num_labels).cuda()
                labels = ones.index_select(0, labels)
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(predicts, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertForSequenceClassificationold(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
class BertForSequenceClassification_bi_gru(BertPreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.num_labels = config.num_labels
        config.lstm_hidden_size = 512
        config.lstm_layers = 1
        config.lstm_dropout = 0.1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.lstm_dropout)
        # self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifier = nn.Linear(config.lstm_hidden_size * 2, self.config.num_labels)

        self.W = []
        self.gru = []

        for i in range(config.lstm_layers):
            self.W.append(nn.Linear(config.lstm_hidden_size * 2, config.lstm_hidden_size * 2))
            self.gru.append(
                nn.GRU(config.hidden_size, config.lstm_hidden_size,
                       num_layers=1, bidirectional=True, batch_first=True).cuda())
        self.W = nn.ModuleList(self.W)
        self.gru = nn.ModuleList(self.gru)
        self.classifier = nn.Linear(config.lstm_hidden_size * 2, self.config.num_labels)
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None


        outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()

        for w, gru in zip(self.W, self.gru):
            gru.flatten_parameters()
            output, hidden = gru(output)
            output = self.dropout(output)

        hidden = hidden.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()

        logits = self.classifier(hidden)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                # loss_fct = FocalLoss(class_num=self.num_labels, alpha=0.25)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)
class BertForSequenceClassification_grulstm(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.lstm_hidden_size = 384
        config.lstm_layers = 1
        config.lstm_dropout = 0.1

        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.lstm_dropout)
        # self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.gru = nn.GRU(config.lstm_hidden_size*2, config.lstm_hidden_size,
               num_layers=1, bidirectional=True, batch_first=True).cuda()
        self.lstm = nn.LSTM(config.hidden_size, config.lstm_hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True).cuda()
        self.classifier = nn.Linear(config.hidden_size*4, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        bert_output = outputs[0]
        pooled_output = outputs[1]

        h_lstm, _ = self.lstm(bert_output)  # [bs, seq, output*dir]
        h_gru, hh_gru = self.gru(h_lstm)    #
        hh_gru = hh_gru.view(-1, 2 * self.config.lstm_hidden_size)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        h_conc_a = torch.cat(
            (avg_pool, hh_gru, max_pool, pooled_output), 1
        )

        output = self.dropout(h_conc_a)
        logits = self.classifier(output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)
        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForSequenceClassification_grulstm(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.lstm_hidden_size = 384
        config.lstm_layers = 1
        config.lstm_dropout = 0.1

        self.config = config
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.lstm_dropout)
        # self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.gru = nn.GRU(config.lstm_hidden_size*2, config.lstm_hidden_size,
               num_layers=1, bidirectional=True, batch_first=True).cuda()
        self.lstm = nn.LSTM(config.hidden_size, config.lstm_hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True).cuda()
        self.classifier = nn.Linear(config.hidden_size*4, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.roberta(input_ids=flat_input_ids, position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        roberta_output = outputs[0]
        pooled_output = outputs[1]

        h_lstm, _ = self.lstm(roberta_output)  # [bs, seq, output*dir]
        h_gru, hh_gru = self.gru(h_lstm)    #
        hh_gru = hh_gru.view(-1, 2 * self.config.lstm_hidden_size)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        h_conc_a = torch.cat(
            (avg_pool, hh_gru, max_pool, pooled_output), 1
        )

        output = self.dropout(h_conc_a)
        logits = self.classifier(output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)
        return outputs  # (loss), logits, (hidden_states), (attentions)