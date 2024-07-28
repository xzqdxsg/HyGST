from tkinter.messagebox import NO
from turtle import forward
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from seq_utils import *
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn,optim
import torch.nn.functional as F
from torchcrf import CRF

from eval_lego_4 import extract_pairs
from glue_utils_seq_gengerate import ABSAProcessor
import pdb

class CRF(nn.Module):
    # borrow the code from 
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
    def __init__(self, num_tags, constraints=None, include_start_end_transitions=None):
        """

        :param num_tags:
        :param constraints:
        :param include_start_end_transitions:
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.include_start_end_transitions = include_start_end_transitions
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        constraint_mask = torch.Tensor(self.num_tags+2, self.num_tags+2).fill_(1.)
        if include_start_end_transitions:
            self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        # register the constraint_mask
        self.constraint_mask = nn.Parameter(constraint_mask, requires_grad=False)
        self.reset_parameters()

    def forward(self, inputs, tags, mask=None):
        """

        :param inputs: (bsz, seq_len, num_tags), logits calculated from a linear layer
        :param tags: (bsz, seq_len)
        :param mask: (bsz, seq_len), mask for the padding token
        :return:
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def reset_parameters(self):
        """
        initialize the parameters in CRF
        :return:
        """
        nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            nn.init.normal_(self.start_transitions)
            nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits, mask):
        """

        :param logits: emission score calculated by a linear layer, shape: (batch_size, seq_len, num_tags)
        :param mask:
        :return:
        """
        bsz, seq_len, num_tags = logits.size()
        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for t in range(1, seq_len):
            # iteration starts from 1
            emit_scores = logits[t].view(bsz, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(bsz, num_tags, 1)

            # calculate the likelihood
            inner = broadcast_alpha + emit_scores + transition_scores

            # mask the padded token when met the padded token, retain the previous alpha
            alpha = (logsumexp(inner, 1) * mask[t].view(bsz, 1) + alpha * (1 - mask[t]).view(bsz, 1))
        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(self, logits, tags, mask):
        """
        calculate the likelihood for the input tag sequence
        :param logits:
        :param tags: shape: (bsz, seq_len)
        :param mask: shape: (bsz, seq_len)
        :return:
        """
        bsz, seq_len, _ = logits.size()

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for t in range(seq_len-1):
            current_tag, next_tag = tags[t], tags[t+1]
            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[t].gather(1, current_tag.view(bsz, 1)).squeeze(1)

            score = score + transition_score * mask[t+1] + emit_score * mask[t]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, bsz)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def viterbi_tags(self, logits, mask):
        """

        :param logits: (bsz, seq_len, num_tags), emission scores
        :param mask:
        :return:
        """
        _, max_seq_len, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = (
                self.transitions * self.constraint_mask[:num_tags, :num_tags] +
                -10000.0 * (1 - self.constraint_mask[:num_tags, :num_tags])
        )

        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                    self.start_transitions.detach() * self.constraint_mask[start_tag, :num_tags].data +
                    -10000.0 * (1 - self.constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                    self.end_transitions.detach() * self.constraint_mask[:num_tags, end_tag].data +
                    -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = (-10000.0 *
                                                 (1 - self.constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_len + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            # perform viterbi decoding sample by sample
            seq_len = torch.sum(prediction_mask)
            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(seq_len + 1), :num_tags] = prediction[:seq_len]
            # And at the last timestep we must have the END_TAG
            tag_sequence[seq_len + 1, end_tag] = 0.
            viterbi_path = viterbi_decode(tag_sequence[:(seq_len + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append(viterbi_path)
        return best_paths


class T5FineTuner(nn.Module):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.T5 = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)


    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        lm_labels = labels
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self.T5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs
    
    def freeze(self,module):
        for parameter in module.parameters():
            parameter.requires_grad = False

class T5encoder(nn.Module):
    def __init__(self, hparams):
        super(T5encoder, self).__init__()
        self.hparams = hparams
        self.num_labels = hparams.num_labels
        # self.lm_head = torch.nn.Linear(768,hparams.num_labels)
        self.T5 = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.crf = CRF(num_tags=hparams.num_labels)
        self.bert_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dense = nn.Linear(768,768)
          
    def forward(self, input_ids, attention_mask=None, labels=None):

        encoder_outputs = self.T5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        outputs = encoder_outputs[0]
        outputs = self.dense(outputs)
        outputs = self.bert_dropout(outputs)
        logits = self.classifier(outputs)
        outputs = (logits,)
        # labels_mask = attention_mask.type(torch.uint8)
        loss = self.crf(logits, labels, attention_mask)
        loss = -1 * loss
        outputs = (loss,) + outputs
        # pdb.set_trace()
        
        return outputs


def feature_extract(com_features, labels,num_labels):
    '''将同一标签对应的特征拼在一起'''
    label_features = {}
    tokens_emb = com_features.view(-1, com_features.size(-1)).clone()
    token_labels = labels.view(-1)
    assert tokens_emb.size(0) == token_labels.size(0)

    # for i in range(num_labels-1):  # 不算'O'标签
    #     tag = i+1
    #     active_indices = torch.where(token_labels==tag)  # 得到标签为tag的索引
    #     label_features[tag] = tokens_emb[active_indices]  # tag对应的特征
    for i in range(num_labels):  # 算'O'标签
        tag = i
        active_indices = torch.where(token_labels==tag)  # 得到标签为tag的索引
        label_features[tag] = tokens_emb[active_indices]  # tag对应的特征
    return label_features

def label_center(com_label_features):
    '''将标签对应的特征合并'''
    labels = list(com_label_features.keys())
    device = com_label_features[1].device
    for k in labels:
        if com_label_features[k].size(0) != 0:
            com_label_features[k] = torch.mean(com_label_features[k],dim=0).unsqueeze(0)
        else:
            com_label_features[k] = torch.zeros(1,768).to(device=device)
    return com_label_features

def label_convert(pseudo_labels, tagging_schema):
    # pl_tag = ['B-POS', 'I-POS', 'B-NEU'...]
    sentiments ={'positive':'POS','negative':'NEG','neutral':'NEU'}
    pl_tag = []
    for label in pseudo_labels:
        if len(label[0]) != 0:
            aspects = label[0].split()
            if tagging_schema == 'BIO':
                    for i in range(len(aspects)):
                        if label[1] in sentiments.keys():
                            if i == 0:
                                pl_tag.append('B-%s' % sentiments[label[1]])
                            else:
                                pl_tag.append('I-%s' % sentiments[label[1]])
            elif tagging_schema == 'OT':
                for i in range(len(aspects)):
                    if label[1] in sentiments.keys():
                        pl_tag.append('T-%s' % sentiments[label[1]])
    return pl_tag

def get_sentence(step, mode, args):
    prompt = 'Find all aspects and sentiment polarity given context . </s> context :'
    prompt_list = prompt.split()
    processor = ABSAProcessor()
    if mode =='train':
        batchsize = args.train_batch_size
    else:
        batchsize = args.eval_batch_size
    
    if mode == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif mode == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    elif mode == 'test':
        examples = processor.get_test_examples(args.data_dir)
    batch = batchsize * step
    batchlist = list(range(batch,batch+batchsize))
    for i, example in enumerate(examples):
        if i in batchlist:
            words = []
            words.extend(prompt_list)
            words.extend(example.text_a.split())


def pl_to_seqids(tokenizer, sentence, pseudo_labels, tagging_schema, max_len, seq_labels):
    
    processor = ABSAProcessor()
    label_list = processor.get_labels(tagging_schema)
    label_map = {label: i for i, label in enumerate(label_list)}
    seq_pl_ids = []
    for i in range(len(pseudo_labels)):
        # pred_list: [('base installation', 'negative'), ('software', 'negative')]
        sentiment = ['positive','negative','neutral']
        seq_pl = []
        pred_list = extract_pairs(pseudo_labels[i])
        ap = []
        for pl in pred_list:
            if pl[1] in sentiment:
                ap.append(pl[0])
        aps = ' '.join(ap)
        pl_ap = aps.split()
        pl_st = label_convert(pred_list, tagging_schema)
        # print(pred_list)
        # print(pl_ap)
        # print(pl_st)
        # print('=========================================================================')
        assert len(pl_ap) == len(pl_st)
        words = sentence[i].split()
        words = [k for k in words if k != '<pad>']
        new_words = []
        for word in words:
            if '.</s>' in word:
                sub_word = word.split('.</s>')
                new_words.append(sub_word[0])
                new_words.append('.')
                new_words.append('</s>')
            elif '!</s>' in word:
                sub_word = word.split('!</s>')
                new_words.append(sub_word[0])
                new_words.append('!')
                new_words.append('</s>')
            elif ',' in word:
                sub_word = word.split(',')
                if "'s" in sub_word:
                    sub_word = word.split("'s")
                    new_words.append(sub_word[0])
                    new_words.append("'s")
                else:
                    new_words.append(sub_word[0])
                new_words.append(',')
            elif "'s" in word:
                sub_word = word.split("'s")
                new_words.append(sub_word[0])
                new_words.append("'s")
            elif "%" in word:
                sub_word = word.split("%")
                new_words.append(sub_word[0])
                new_words.append("%")
            elif "n't" in word:
                sub_word = word.split("n't")
                new_words.append(sub_word[0])
                new_words.append("n't")
            elif "'ve" in word:
                sub_word = word.split("'ve")
                new_words.append(sub_word[0])
                new_words.append("'ve")
            elif "'m" in word:
                sub_word = word.split("'m")
                new_words.append(sub_word[0])
                new_words.append("'m")
            elif ":!" in word:
                sub_word = word.split("!")
                new_words.append(sub_word[0])
                new_words.append("!")

            else:
                new_words.append(word)
        new_sentence = ' '.join(new_words)
        new_words = new_sentence.split()
        # print(new_words)
        for word in new_words:
            if word in pl_ap:
                ids = pl_ap.index(word)
                seq_pl.append(pl_st[ids])
            else:
                seq_pl.append('O')
        tokenizer_pl = []
        tokenizer_words = []
        for word, p_label in zip(new_words, seq_pl):
            subwords = tokenizer.tokenize(word)
            tokenizer_words.extend(subwords)
            if p_label != 'O':
                tokenizer_pl.extend([p_label] + ['EQ'] *(len(subwords) - 1))
            else:
                tokenizer_pl.extend(['O'] * len(subwords))
        if len(tokenizer_pl) < max_len:
            tokenizer_pl.extend(['O']*(max_len-len(tokenizer_pl)))
        seq_pseudo_labels = [label_map[label] for label in tokenizer_pl]
        seq_pl_ids.append(seq_pseudo_labels)
        # if seq_labels[i].tolist()!=seq_pseudo_labels:
        #     pdb.set_trace()
    seq_pl_ids= torch.tensor([ids for ids in seq_pl_ids], dtype=torch.long)
    return seq_pl_ids

class FeatureMerge(nn.Module):
    def __init__(self):
        super(FeatureMerge, self).__init__()
        self.merge = nn.Linear(768 * 2, 768)
        self.dense = nn.Linear(768, 768)

    def forward(self, com_features, labels, label_feature):
        # with torch.no_grad():
        bsz = com_features.size(0)
        hidden_size = com_features.size(-1)
        tokens_emb = com_features.view(-1, hidden_size)
        token_embedding = tokens_emb#.clone()
        token_labels = labels.view(-1)
        assert tokens_emb.size(0) == token_labels.size(0)
        active_indice = torch.where(token_labels!=0)  # 得到伪标签不为'O'的token的索引
        active_indice = active_indice[0]
        all_label_features = label_feature[1]
        keys = list(label_feature.keys())
        for k in keys:
            if k > 1:
                all_label_features = torch.cat((all_label_features,label_feature[k]), dim=0)  # 维度0-6,对应标签1-7
        token_label_emb = torch.index_select(token_embedding, dim=0, index=active_indice)  # 根据标签索引取出所有标签 不为'O'的token特征
        token_label_label = torch.index_select(token_labels, dim=0, index=active_indice)  # 根据标签索引取出所有标签不为'O'的标签，即上述token的对应标签
        label_id = token_label_label.add(-1)  # 标签值减1，0-6对应标签1-7
        token_label_center = torch.index_select(all_label_features, dim=0, index=label_id)  # 根据标签值取标签特征

        if token_label_emb.size(0) != 0:
            com_token_label_features = torch.cat((token_label_emb,token_label_center), dim=-1)
            com_token_label_features = self.merge(com_token_label_features)

        else:
            com_token_label_features = token_label_emb
        indices = active_indice.unsqueeze(1).expand(active_indice.size(0),hidden_size)
        # token_embedding = tokens_emb.clone()
        tokens_emb_update = token_embedding.scatter_(dim=0, index=indices, src=com_token_label_features)
        com_features_update = tokens_emb_update.view(bsz,-1,hidden_size)
        # pdb.set_trace()
        return com_features_update

class T5_generate(nn.Module):
    def __init__(self, hparams):
        super(T5_generate, self).__init__()
        self.hparams = hparams
        # self.num_labels = hparams.num_labels
        # self.attention_heads = hparams.attention_heads
        # self.lm_head = torch.nn.Linear(768,hparams.num_labels)
        self.T5 = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        # self.T5encoder = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path).get_encoder()
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        # self.merge = FeatureMerge()
        # self.layers = 1
        # self.attn =ConsineMultiHeadAttention(self.hparams, self.attention_heads, 768, h_dim=self.hparams.h_dim)
        # self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # self.weight = nn.ModuleList([nn.Linear(768,768) for _ in range(self.layers)])
        # self.crf = CRF(num_tags=hparams.num_labels)
        # self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(768, self.num_labels)
        # self.dense = nn.Linear(768,768)
        # self.loss = nn.CrossEntropyLoss(ignore_index=-100)
          
    def forward(self, input_ids, attention_mask=None, seq_labels=None, generate_label_ids=None, generate_label_mask=None,
                mode='train', step=0, labels_feature=None):
        
        
        lm_labels = generate_label_ids
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.T5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            decoder_attention_mask=generate_label_mask
        )
        # encoder_outputs = outputs.encoder_last_hidden_state

        
        return outputs, labels_feature
class T5encoder_decoder(nn.Module):
    def __init__(self, hparams):
        super(T5encoder_decoder, self).__init__()
        self.hparams = hparams
        self.num_labels = hparams.num_labels
        self.attention_heads = hparams.attention_heads
        # self.lm_head = torch.nn.Linear(768,hparams.num_labels)
        self.T5 = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        # self.T5encoder = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path).get_encoder()
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        # self.merge = FeatureMerge()
        self.layers = 1
        self.attn =ConsineMultiHeadAttention(self.hparams, self.attention_heads, 768, h_dim=self.hparams.h_dim)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.weight = nn.ModuleList([nn.Linear(768,768) for _ in range(self.layers)])
        self.crf = CRF(num_tags=hparams.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dense = nn.Linear(768,768)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
          
    def forward(self, input_ids, attention_mask=None, seq_labels=None, generate_label_ids=None, generate_label_mask=None,
                mode='train', step=0, labels_feature=None):
        
        
        lm_labels = generate_label_ids
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.T5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            decoder_attention_mask=generate_label_mask
        )
        # encoder_outputs = outputs.encoder_last_hidden_state
        generate_loss = outputs.loss

        # encoder_outputs = self.T5.encoder(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask
        # )
        # encoder_outputs = encoder_outputs[0]
        # com_encoder_feature = encoder_outputs.detach()
        # com_label_features = feature_extract(com_encoder_feature, seq_labels, self.num_labels)
        # if mode != 'train':
        #     labels_feature = labels_feature
        # elif step == 0:
        #     labels_feature = label_center(com_label_features)
        # else:
        #     keys = list(com_label_features.keys())
        #     for k in keys:
        #         com_label_features[k] = torch.cat((labels_feature[k],com_label_features[k]),dim=0)
        #     labels_feature = label_center(com_label_features)
        # if mode == 'train':
        #     seq_pl_ids = seq_labels
        # else:
        #     max_len = input_ids.size(1)
        #     generate_outs = self.T5.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        #     pseudo_labels = [self.tokenizer.decode(ids) for ids in generate_outs]
        #     sentence = [self.tokenizer.decode(ids) for ids in input_ids]
        #     seq_pl_ids = pl_to_seqids(self.tokenizer, sentence, pseudo_labels, self.hparams.tagging_schema, max_len, seq_labels)
        ## 融合伪标签
        # keys = list(com_label_features.keys())
        # for k in keys:
        #     labels_feature[k] = labels_feature[k].to(input_ids.device)
        # seq_outputs = self.merge(encoder_outputs, seq_pl_ids.to(input_ids.device), labels_feature)
        
        # GCN

        # device = input_ids.device
        # seq_pl_ids = seq_pl_ids.to(device)
        # ## 注意力GCN
        # ## 合并标签中心
        # all_label_features = labels_feature[1]
        # keys = list(labels_feature.keys())[1:]
        # for k in keys:
        #     if k > 1:
        #         all_label_features = torch.cat((all_label_features,labels_feature[k]), dim=0)

        # all_label_features = all_label_features.unsqueeze(0).repeat(input_ids.size(0),1,1).to('cpu')  # [32,8,768]
        # all_label_features = all_label_features.detach()
        # seq_features = encoder_outputs.to('cpu')

        # att_inputs = torch.cat((all_label_features, seq_features), dim=1).to(device)  # [32,112,768]
        # label_center_mask = torch.ones((input_ids.size(0),len(keys)), dtype=int, device=device)
        # attmask = torch.cat((label_center_mask, attention_mask), dim=-1)
        # attmask = attmask.unsqueeze(-1).repeat(1,1,attmask.size(1))
        # label_center_label = torch.tensor(keys, dtype=int, device=device)
        # label_center_label = label_center_label.unsqueeze(0).repeat(input_ids.size(0),1)
        # att_generate_ids = torch.cat((label_center_label, seq_pl_ids), dim=-1)
        # att_generate_ids = att_generate_ids + 1     

        # ## attention_score
        # adj_ag = self.attn(att_inputs, att_inputs, attmask)
        # ids_score = torch.zeros((att_generate_ids.size(0),att_generate_ids.size(1),att_generate_ids.size(1)),device=device)
        # # pdb.set_trace()
        # for j in range(ids_score.size(0)):
        #     ## attgcn
        #     adj_ag[j] = adj_ag[j] - torch.diag(torch.diag(adj_ag[j]))  # 将对角线元素置为0
        #     adj_ag[j] = adj_ag[j] + torch.eye(adj_ag[j].size(0),device=device)  # 将对角线元素变成 1
        #     ids_row = att_generate_ids[j].unsqueeze(0)
        #     ids_column = (1.0 / att_generate_ids[j])
        #     ids_column = torch.cat((ids_column[:len(keys)], torch.where(ids_column[len(keys):]==1.0, torch.zeros(generate_label_ids.size(1), device=device), ids_column[len(keys):])),dim=0)
        #     ids_column = ids_column.unsqueeze(1)
        #     ids_score[j] = ids_row * ids_column
        #     ##
        #     ids_score[j] = ids_score[j] - torch.diag(torch.diag(ids_score[j]))
        #     # ids_score[j] = ids_score[j] + torch.eye(ids_score[j].size(0),device=device)
        #     ##
        #     replacement = torch.zeros_like(ids_score[j])
        #     ids_score[j] = torch.where(ids_score[j] != 1.0, replacement, ids_score[j])
        #     ##
        #     # ids_score[j][len(keys):, len(keys):] = 0
        #     # pdb.set_trace()

        #     old_score = adj_ag[j] + ids_score[j]
        #     old_score = torch.where(old_score<=1.0, replacement, old_score)
        #     old_score = old_score - ids_score[j]
        #     adj_ag[j] = adj_ag[j] + ids_score[j]
        #     adj_ag[j] = adj_ag[j] - old_score


        # denom_ag = ids_score.sum(2).unsqueeze(2) + 1
        # Ax = ids_score.bmm(att_inputs)
        # gAxW = att_inputs
        # for weight in self.weight:
        #     denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        #     Ax = adj_ag.bmm(gAxW)
        #     AxW = weight(Ax)
        #     AxW = AxW / denom_ag
        #     gAxW = F.relu(AxW)

        # gcn_outputs = gAxW[:, len(keys):, :]

        # T5encoder -> Sequence Tagging
        
        # outputs = gcn_outputs

        # # # T5encoder -> Sequence Tagging
        # outputs = self.dense(outputs)
        # outputs = self.dropout(outputs)
        # logits = self.classifier(outputs)
        # outputs = (logits,)
        # # labels_mask = attention_mask.type(torch.uint8)
        # seqloss = self.crf(logits, seq_labels, attention_mask)
        # seqloss = -1 * seqloss
        # loss = 0.01*seqloss + generate_loss
        loss = generate_loss
        # loss = seqloss
        # outputs = (loss,) + outputs
        # pdb.set_trace()
        
        return outputs, labels_feature

class ConsineMultiHeadAttention(nn.Module):

    def __init__(self, args, h, d_model, h_dim=120, dropout=0.1):
        super(ConsineMultiHeadAttention, self).__init__()
        self.args = args
        assert d_model % h == 0
        assert d_model % (2*h) == 0

        self.d_k = h_dim * h
        self.h_dim = h_dim
        self.h = h
        self.q_linear = nn.Linear(d_model, self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.dropout = nn.Dropout(p=dropout)

        self.weight_tensor = torch.Tensor(h, self.d_k)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
        
        nbatches = query.size(0)

        context = self.q_linear(query).view(nbatches, -1, self.h_dim)  # torch.Size([32, 112, 60])
        
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        # if len(context.shape) == 3:
        #     expand_weight_tensor = expand_weight_tensor.unsqueeze(1)  # torch.Size([1, 1, 60])

        context_fc = context * expand_weight_tensor  # torch.Size([32, 112, 60])
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        scores = torch.matmul(context_norm, context_norm.transpose(-1, -2))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return p_attn


def get_cosine_dis(input_0, label_len, mask):
    bsz = input_0.size(0)
    seq_len = input_0.size(1)
    dim = input_0.size(2)
    # input_0 = F.normalize(input_0.detach(), p=2, dim=-1)
    for i in range(bsz):
        x = input_0[i].unsqueeze(0).expand(seq_len,seq_len,dim)
        y = input_0[i].unsqueeze(1).expand(seq_len,seq_len,dim)
        score = F.cosine_similarity(x, y, dim=-1)
        if i == 0:
            sc_score = score.unsqueeze(0)
        else:
            sc_score = torch.cat((sc_score,score.unsqueeze(0)),dim=0)
    max_value = torch.max(sc_score)
    min_value = torch.min(sc_score)
    sc_score = (sc_score - min_value) / (max_value - min_value)
    sc_score[:,:label_len,:label_len] = 0
    # sc_score[:,label_len:,label_len:] = 0
    # sc_score = F.normalize(sc_score, dim=tuple(range(sc_score.dim())))
    # pdb.set_trace()

    if mask is not None:
            sc_score = sc_score.masked_fill(mask == 0, -1e9)

    return sc_score


class T5encoder_merge(nn.Module):
    def __init__(self, hparams):
        super(T5encoder_merge, self).__init__()
        self.hparams = hparams
        self.num_labels = hparams.num_labels
        self.attention_heads = hparams.attention_heads
        # self.lm_head = torch.nn.Linear(768,hparams.num_labels)
        # self.T5 = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.T5encoder = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path).get_encoder()
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.merge = FeatureMerge()
        # self.layers = 3
        self.attn =ConsineMultiHeadAttention(self.hparams, self.attention_heads, 768, h_dim=self.hparams.h_dim)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # self.weight = nn.ModuleList([nn.Linear(768,768) for _ in range(self.layers)])
        self.weight = nn.Linear(768,768)
        self.crf = CRF(num_tags=hparams.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dense = nn.Linear(768,768)
        self.label_emb = nn.Embedding(self.num_labels, 768)
        # self.linear = nn.Linear(768,768)

          
    def forward(self, input_ids, attention_mask=None, seq_labels=None, generate_label_ids=None,
                step=0, labels_feature=None):
        encoder_outputs = self.T5encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_outputs = encoder_outputs[0]

        com_label_features = feature_extract(encoder_outputs, seq_labels, self.num_labels)
        if step == -1:
            labels_feature = labels_feature
        elif step == 0:
            labels_feature = label_center(com_label_features)
        else:
            keys = list(com_label_features.keys())
            for k in keys:
                com_label_features[k] = torch.cat((labels_feature[k],com_label_features[k]),dim=0)
            labels_feature = label_center(com_label_features)

        ## 融合伪标签
        # keys = list(com_label_features.keys())
        # for k in keys:
        #     labels_feature[k] = labels_feature[k].to(input_ids.device).detach()
        # seq_outputs = self.merge(encoder_outputs, generate_label_ids, labels_feature)  #[32,104,768]

        ## GCN

        device = input_ids.device

        ## 注意力GCN
        ## 合并标签中心
        all_label_features = labels_feature[1]
        keys = list(labels_feature.keys())[1:]
        for k in keys:
            if k > 1:
                all_label_features = torch.cat((all_label_features,labels_feature[k]), dim=0)

        all_label_features = all_label_features.unsqueeze(0).repeat(input_ids.size(0),1,1).to('cpu')  # [32,8,768]
        all_label_features = all_label_features.detach()
        seq_features = encoder_outputs.to('cpu')
        # seq_features = seq_outputs.to('cpu')
        ## 将标签与句子cat
        # all_label_features = self.linear(all_label_features).to('cpu')


        ## label Embedding：nn.embedding
        # keys = [1,2,3,4]
        # rand_label_features = torch.LongTensor(keys,device='cpu').to(device)
        # rand_label_features = self.label_emb(rand_label_features)
        # rand_label_features = rand_label_features.unsqueeze(0).repeat(input_ids.size(0),1,1).to('cpu')

        # all_label_features = all_label_features + rand_label_features
        
        att_inputs = torch.cat((all_label_features, seq_features), dim=1).to(device)  # [32,112,768]
        label_center_mask = torch.ones((input_ids.size(0),len(keys)), dtype=int, device=device)
        attmask = torch.cat((label_center_mask, attention_mask), dim=-1)
        attmask = attmask.unsqueeze(-1).repeat(1,1,attmask.size(1))
        label_center_label = torch.tensor(keys, dtype=int, device=device)
        label_center_label = label_center_label.unsqueeze(0).repeat(input_ids.size(0),1)
        att_generate_ids = torch.cat((label_center_label, generate_label_ids), dim=-1)
        att_generate_ids = att_generate_ids + 1     

        ## attention_score
        adj_ag = self.attn(att_inputs, att_inputs, attmask)
        ids_score = torch.zeros((att_generate_ids.size(0),att_generate_ids.size(1),att_generate_ids.size(1)),device=device)
        # pdb.set_trace()
        for j in range(ids_score.size(0)):
            ## attgcn
            adj_ag[j] = adj_ag[j] - torch.diag(torch.diag(adj_ag[j]))  # 将对角线元素置为0
            adj_ag[j] = adj_ag[j] + torch.eye(adj_ag[j].size(0),device=device)  # 将对角线元素变成 1
            ids_row = att_generate_ids[j].unsqueeze(0)
            ids_column = (1.0 / att_generate_ids[j])
            ids_column = torch.cat((ids_column[:len(keys)], torch.where(ids_column[len(keys):]==1.0, torch.zeros(generate_label_ids.size(1), device=device), ids_column[len(keys):])),dim=0)
            ids_column = ids_column.unsqueeze(1)
            ids_score[j] = ids_row * ids_column
            ##
            ids_score[j] = ids_score[j] - torch.diag(torch.diag(ids_score[j]))
            # ids_score[j] = ids_score[j] + torch.eye(ids_score[j].size(0),device=device)
            ##
            replacement = torch.zeros_like(ids_score[j])
            ids_score[j] = torch.where(ids_score[j] != 1.0, replacement, ids_score[j])
            ##
            # ids_score[j][len(keys):, len(keys):] = 0
            # pdb.set_trace()

            old_score = adj_ag[j] + ids_score[j]
            old_score = torch.where(old_score<=1.0, replacement, old_score)
            old_score = old_score - ids_score[j]
            adj_ag[j] = adj_ag[j] + ids_score[j]
            adj_ag[j] = adj_ag[j] - old_score


        # denom_ag = ids_score.sum(2).unsqueeze(2) + 1
        # Ax = ids_score.bmm(att_inputs)
        gAxW = att_inputs
        # for weight in self.weight:
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        Ax = adj_ag.bmm(gAxW)
        AxW = self.weight(Ax)
        AxW = AxW / denom_ag
        gAxW = F.relu(AxW)

        gcn_outputs = gAxW[:, len(keys):, :]

        # T5encoder -> Sequence Tagging
        
        outputs = gcn_outputs
        outputs = self.dense(outputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)

        logit = (logits,)
        # labels_mask = attention_mask.type(torch.uint8)
        seqloss = self.crf(logits, seq_labels, attention_mask)
        loss = -1 * seqloss
        outputs = (loss,) + logit
        # pdb.set_trace()
        
        return outputs, labels_feature

class Postprocess(nn.Module):
    def __init__(self, hparams):
        super(Postprocess, self).__init__()
        self.hparams = hparams
        self.lm_head = torch.nn.Linear(768,hparams.num_labels)
        self.dropout = torch.nn.Dropout(0.1)
        self.crf = CRF(num_tags=hparams.num_labels, batch_first=True)
    
    def forward(self, encoder_output, labels, labels_mask):
        outputs = self.dropout(encoder_output)
        logits = self.lm_head(outputs)
        labels_mask = labels_mask.type(torch.uint8)
        loss = self.crf(logits,labels,labels_mask)
        loss = -1 * loss
        pdb.set_trace()
        return outputs

class T5decoder(nn.Module):
    def __init__(self, hparams):
        super(T5decoder, self).__init__()
        self.hparams = hparams
        self.T5 = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self,input_ids, attention_mask,decoder_ids,decoder_attention_mask,encoder_output):

        lm_labels = decoder_ids
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        decoder_input_ids = self.T5._shift_right(lm_labels)
        decoder_outputs = self.T5.decoder(input_ids=decoder_input_ids,
                                          attention_mask=decoder_attention_mask,
                                          encoder_hidden_states=encoder_output[0],
                                          encoder_attention_mask=attention_mask)
        decoder_outputs = self.T5.lm_head(decoder_outputs.last_hidden_state)
        loss = None
        if decoder_ids is not None:
            loss = self.loss(decoder_outputs.view(-1, decoder_outputs.size(-1)),decoder_ids.view(-1))

        return loss
