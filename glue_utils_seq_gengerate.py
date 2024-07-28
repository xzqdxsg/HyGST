import csv
import logging
import os
import sys
from io import open
import numpy as np

from seq_utils import *

logger = logging.getLogger(__name__)

SMALL_POSITIVE_CONST = 1e-4

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, seq_label=None, generate_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.seq_label = seq_label
        self.generate_label = generate_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class SeqInputFeatures(object):
    """A single set of features of data for the ABSA task"""
    def __init__(self, input_ids, input_mask, seq_label_ids, generate_label_ids, generate_label_mask, evaluate_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seq_label_ids = seq_label_ids
        self.generate_label_ids = generate_label_ids
        self.generate_label_mask = generate_label_mask
        # mapping between word index and head token index
        self.evaluate_label_ids = evaluate_label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class ABSAProcessor(DataProcessor):
    """Processor for the ABSA datasets"""
    def get_train_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='train', tagging_schema=tagging_schema)

    def get_dev_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='dev', tagging_schema=tagging_schema)

    def get_test_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='test', tagging_schema=tagging_schema)

    def get_labels(self, tagging_schema):
        if tagging_schema == 'BIO':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'OT':
            return ['O','EQ', 'T-POS', 'T-NEG', 'T-NEU']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)

    def _create_examples(self, data_dir, set_type, tagging_schema):
        examples = []
        senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
        file = os.path.join(data_dir, "%s.txt" % set_type)
        class_count = np.zeros(3)
        with open(file, 'r', encoding='UTF-8') as fp:
            sample_id = 0
            for i,line in enumerate(fp):
                sent_string, tag_string = line.strip().split('####')
                words = sent_string.split()
                seq_labels = ['O'] * len(words)
                generate_labels = []
                # tup: ([3, 4], POS)
                tuples = eval(tag_string)
                if tuples != []:
                    # tup: ([3, 4], POS)
                    for tup in tuples:
                        ap, sentiment = tup[0], tup[1]
                        for ap_i in range(ap[0],ap[-1]+1):
                            if tagging_schema == 'BIO':
                                if ap_i == ap[0]:
                                    seq_labels[ap_i] = 'B-{}'.format(sentiment)
                                else:
                                    seq_labels[ap_i] = 'I-{}'.format(sentiment)
                            elif tagging_schema == 'OT':
                                seq_labels[ap_i] = 'T-{}'.format(sentiment)
                            generate_labels.append(words[ap_i])
                        generate_labels.append(':')
                        generate_labels.append(senttag2word[sentiment])
                        generate_labels.append('.')
                    generate_labels.pop()

                guid = "%s-%s" % (set_type, sample_id)
                text_a = ' '.join(words)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, seq_label=seq_labels, generate_label=generate_labels))
                sample_id += 1
        return examples


def convert_examples_to_seq_features(examples, label_list, tokenizer,
                                     pad_token=0, sequence_a_segment_id=0,
                                     mask_padding_with_zero=True):
    # feature extraction for sequence labeling
    prompt = 'Find all aspects and sentiment polarity given context . </s> context :'
    prompt_list = prompt.split()
    len_p = 0
    prompt_tokenizer = []
    prompt_labels = []
    for p in prompt_list:
        sub_p = tokenizer.tokenize(p)
        len_p += len(sub_p)
        prompt_tokenizer.extend(sub_p)
        prompt_labels.extend(['O']*len(sub_p))

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_seq_length = -1
    examples_tokenized = []

    for (ex_index, example) in enumerate(examples):
        tokens_a = []
        seq_labels_a = []
        evaluate_label_ids = []
        words = []
        words.extend(prompt_list)
        words.extend(example.text_a.split())
        seq_labels = []
        seq_labels.extend(['O']*len(prompt_list))
        seq_labels.extend(example.seq_label)
        generate_labels = ' '.join(example.generate_label)
        # pdb.set_trace()
        # words = example.text_a.split(' ')
        # labels = example.label
        wid, tid = 0, 0
        for word, seqlabel in zip(words, seq_labels):
            subwords = tokenizer.tokenize(word)
            tokens_a.extend(subwords)
            if seqlabel != 'O':
                seq_labels_a.extend([seqlabel] + ['EQ'] * (len(subwords) - 1))
            else:
                seq_labels_a.extend(['O'] * len(subwords))
            evaluate_label_ids.append(tid)
            wid += 1
            # move the token pointer
            tid += len(subwords)

        #print(evaluate_label_ids)
        assert tid == len(tokens_a)

        # tokens_a = prompt_tokenizer + tokens_a
        # labels_a = prompt_labels + labels_a

        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        examples_tokenized.append((words, tokens_a, seq_labels_a, generate_labels, evaluate_label_ids))
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    # count on the [CLS] and [SEP]
    max_seq_length += 2
    #max_seq_length = 128
    for ex_index, (words, tokens_a, seq_labels_a, generate_labels, evaluate_label_ids) in enumerate(examples_tokenized):
        #tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        # for sequence labeling, better not truncate the sequence
        #if len(tokens_a) > max_seq_length - 2:
        #    tokens_a = tokens_a[:(max_seq_length - 2)]
        #    labels_a = labels_a
        words = words + ['</s>']
        tokens = tokens_a + ['</s>']
        segment_ids = [sequence_a_segment_id] * len(tokens)
        seq_labels = seq_labels_a + ['O']
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # input_mask[:len_p] = [0] * len_p

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        #print("Current labels:", labels)
        seq_label_ids = [label_map[label] for label in seq_labels]

        # pad the input sequence and the mask sequence
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        # pad sequence tag 'O'
        seq_label_ids = seq_label_ids + ([0] * padding_length)

        generate_label_all = tokenizer.batch_encode_plus(
            [generate_labels], max_length = max_seq_length, pad_to_max_length=True, truncation=True,
            return_tensors="pt"
        )
        generate_label_ids = generate_label_all['input_ids'].tolist()[0]
        generate_label_mask = generate_label_all['attention_mask'].tolist()[0]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(seq_label_ids) == max_seq_length
        assert len(generate_label_ids) == max_seq_length
        assert len(generate_label_mask) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels: %s " % ' '.join([str(x) for x in seq_label_ids]))
            logger.info("evaluate label ids: %s" % evaluate_label_ids)

        features.append(
            SeqInputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             seq_label_ids=seq_label_ids,
                             generate_label_ids=generate_label_ids,
                             generate_label_mask = generate_label_mask,
                             evaluate_label_ids=evaluate_label_ids))
    print("maximal sequence length is", max_seq_length)
    return features, len_p


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        #print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count


def compute_metrics_absa(preds, labels, all_evaluate_label_ids, tagging_schema,args, mode=None, prompt_len=0):
    Falselabel_file_path = '%s/%s/%s_false.txt' % (args.output_dir, args.checkpoint, mode)
    Falselabel_file = open(Falselabel_file_path, 'w')


    if tagging_schema == 'BIEOS':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5,
                        'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9,
                        'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
    elif tagging_schema == 'BIO':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 
        'B-NEG': 4, 'I-NEG': 5, 'B-NEU': 6, 'I-NEU': 7}
    elif tagging_schema == 'OT':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
    else:
        raise Exception("Invalid tagging schema %s..." % tagging_schema)
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    # precision, recall and f1 for aspect-based sentiment analysis
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    n_samples = len(all_evaluate_label_ids)
    pred_y, gold_y = [], []
    class_count = np.zeros(3)
    false_num = 0
    slip_num = 0
    sentiment_false = 0
    for i in range(n_samples):
        # labels_i = labels[i][prompt_len:]
        evaluate_label_ids = all_evaluate_label_ids[i]
        pred_labels = preds[i][evaluate_label_ids]
        # gold_labels = labels_i[evaluate_label_ids]
        gold_labels = labels[i][evaluate_label_ids]
        assert len(pred_labels) == len(gold_labels)
        # here, no EQ tag will be induced
        pred_tags = [absa_id2tag[label] for label in pred_labels]
        gold_tags = [absa_id2tag[label] for label in gold_labels]

        if tagging_schema == 'OT':
            gold_tags = ot2bieos_ts(gold_tags)
            pred_tags = ot2bieos_ts(pred_tags)
        elif tagging_schema == 'BIO':
            gold_tags = ot2bieos_ts(bio2ot_ts(gold_tags))
            pred_tags = ot2bieos_ts(bio2ot_ts(pred_tags))
        else:
            # current tagging schema is BIEOS, do nothing
            pass
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)

        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)
        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        for (b, e, s) in g_ts_sequence:
            if s == 'POS':
                class_count[0] += 1
            if s == 'NEG':
                class_count[1] += 1
            if s == 'NEU':
                class_count[2] += 1
        aps_pred = []
        aps_gold = []
        for k in range(len(p_ts_sequence)):
            aps_pred.append(p_ts_sequence[k][:2])
        for k in range(len(g_ts_sequence)):
            aps_gold.append(g_ts_sequence[k][:2])
        for tseq in g_ts_sequence:
            for pseq in p_ts_sequence:
                if tseq[:2] == pseq[:2] and tseq[2] != pseq[2]:
                    sentiment_false += 1


        for tup in aps_pred:
            if tup not in aps_gold:
                false_num += 1
        for tup in aps_gold:
            if tup not in aps_pred:
                slip_num += 1

        if p_ts_sequence != g_ts_sequence:
            Falselabel_file.write("id: {}  pred labels:{}  gold labels:{}. \n".format(i, p_ts_sequence, g_ts_sequence))
    Falselabel_file.write("错误提取方面个数: {}  少提方面个数:{} . \n".format(false_num, slip_num))

    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    # TP
    n_tp_total = sum(n_tp_ts)
    # TP + FN
    n_g_total = sum(n_gold_ts)
    print("class_count:", class_count)

    # TP + FP
    n_p_total = sum(n_pred_ts)
    micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + SMALL_POSITIVE_CONST)
    scores = {'macro-f1': macro_f1, 'precision': micro_p, "recall": micro_r, "micro-f1": micro_f1}
    Falselabel_file.write("预测正确个数: {}  全部预测个数:{}  真实标签个数{} . \n".format(n_tp_total, n_p_total, n_g_total))
    return scores
