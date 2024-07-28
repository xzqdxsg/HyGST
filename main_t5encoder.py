import argparse
import os
import torch
import torch.nn.functional as F
import logging
from logging import FileHandler
import random
import numpy as np
import sys
from tqdm import tqdm, trange

from glue_utils_merge import convert_examples_to_seq_features, ABSAProcessor, compute_metrics_absa
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from T5FineTuner import T5encoder_merge
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pdb

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: [bert,t5]")
    parser.add_argument("--absa_type", default=None, type=str, 
                        help="Downstream absa layer type selected in the list: [linear, gru, san, tfm, crf]")
    parser.add_argument("--tfm_mode", default=None, type=str,
                        help="mode of the pre-trained transformer, selected from: [finetune]")
    parser.add_argument("--checkpoint", default=None, type=str, required=True,
                        help="save model address")
    parser.add_argument("--fix_tfm", default=None, type=int,
                        help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: [laptop14, rest14, rest15, rest16]")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_tsne", action='store_true',
                        help="Whether to run tsne on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--tagging_schema', type=str, default='BIO')

    parser.add_argument("--overfit", type=int, default=0, help="if evaluate overfit or not")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--attention_heads', default=1, type=int)   # 4
    parser.add_argument('--h_dim', default=60, type=int)  # 120   60
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()
    output_dir = '%s-%s' % (args.model_type, args.task_name)
    args.output_dir = output_dir
    return args

def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # draw training samples from shuffled dataset
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    dev_best_f1 = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # set the seed number
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    epoch = 0
    for _ in train_iterator:
        labels_feature = None
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':          batch[0],
                      'attention_mask':     batch[1],
                      'seq_labels':         batch[2],
                      'generate_label_ids': batch[3],
                      'step':               step,
                      'labels_feature':     labels_feature,}
            outputs,labels_feature = model(**inputs)
            # loss with attention mask
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            global_step += 1

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint per each N steps
                # print('Training complete in {:.0f}m {:.0f}s'.format(time_all // 60, time_all % 60))
                output_dir = os.path.join(args.output_dir, args.checkpoint)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                train_result = evaluate(args, model, tokenizer, labels_feature, mode='train')
                dev_result = evaluate(args, model, tokenizer, labels_feature, mode='dev')
                test_result = evaluate(args, model, tokenizer, labels_feature, mode='test')

                logger.info(
                    "step: {} Train f1: {:.4f}, Train loss: {:.4f}, dev_f1: {:.4f}, dev loss: {:.4f}, Test f1: {:.4f}, Test loss: {:.4f}."
                    .format(global_step, train_result['micro-f1'], train_result['eval_loss'], dev_result['micro-f1'], dev_result['eval_loss'], test_result['micro-f1'], test_result['eval_loss'])
                    )
                if dev_result['micro-f1'] > dev_best_f1:
                    logger.info("===================new checkpoint==================")
                    dev_best_f1 = dev_result['micro-f1']

                    torch.save(model.state_dict(), os.path.join(output_dir, 'model_parameter.pkl'))
                    torch.save(labels_feature,os.path.join(output_dir, 'labels_feature.pth'))
                    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def hotmap(logits):
    plt.rcParams['font.family'] = 'serif'
    # len_sentence = ['This', 'place', 'is', 'Italian', '▁', ',', 'not', 'French', '.']
    # len_sentence = ['▁it', '▁is', '▁', 'a', '▁cozy', '▁place', '▁to', '▁go', '▁with', '▁', 'a', '▁couple', '▁of', '▁friends', '▁', '.', '</s>']
    # len_sentence = ['Chef', 'Wald', 'y', '▁', "'", 's', 'always', 'measures', 'up', '.']
    # len_sentence = ['▁The', '▁food', '▁is', '▁all', '-', 'around', '▁good', '▁', ',', '▁with', '▁the', '▁rolls', '▁usually', '▁excellent', '▁and', '▁the', '▁sushi', '/', 's', 'ash', 'imi', '▁not', '▁quite', '▁on', '▁the', '▁same', '▁level', '▁', '.', '</s>']
    len_sentence = ['Die', 'ters', 'stick', 'to', 'salad', 's', 'or', 'indulge', 'in', 'vegetarian', 'platter', 's']
    # len_sentence = ['▁Have', '▁', 'a', '▁mo', 'jit', 'o', '▁and', '▁sit', '▁in', '▁the', '▁back', '▁patio', '▁', '.', '</s>']
    # len_sentence = ['▁I', '▁would', '▁go', '▁back', '▁for', '▁the', '▁wine', '▁experience', '▁alone', '▁', '.', '</s>']
    # len_sentence = ['The', 'place', 'is', 'the', 'next', 'best', 'thing', 'to', 'my', 'Mom', 's', 'cooking', '.']
    tags = ['O','EQ', 'T-POS', 'T-NEG', 'T-NEU']
    # pdb.set_trace()
    logits = torch.softmax(logits,dim=2)
    logits = torch.log10(logits)
    data = torch.cat((logits[4][:len(len_sentence)-1],logits[4][len(len_sentence)+1].unsqueeze(0)), dim=0).detach().cpu().numpy()

    # data = logits[4][:len(len_sentence)+1].detach().cpu().numpy()
    data = data.transpose()
    # data = np.log(data)
    # 创建 DataFrame

    df = pd.DataFrame(data, columns=list(len_sentence), index=list(tags))
    
    # 制作热力图
    fig, ax = plt.subplots(figsize=(8,8))
    # plt.figure(figsize=(12, 6))
    # sns.set(font_scale=1.2)  # 设置字体比例
    # custom_colors = sns.diverging_palette(20, 220, n=256, as_cmap=True)  # 创建自定义的调色板
    # custom_colors = ["#8B4513", "#FF4500"]  # 棕色和红色
    # 设置横轴和纵轴标签
    # plt.xlabel('Tags', fontsize=40)  # 设置标签字体大小
    # plt.ylabel('Sentence', fontsize=40)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=15)
    heatmap = sns.heatmap(df, annot=False, cmap='YlGnBu',cbar=False,annot_kws={"fontsize":10}, fmt='.1f', square=True)  # 保留一位小数
    # cbar = heatmap.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=13)
    # heatmap = sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.subplots_adjust(left=0.02,right=1.05)
    ax.tick_params(axis='x', length=5)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    # 添加标题
    # plt.title('Heatmap', fontsize=20)
    
    # plt.show()
    plt.savefig('Heatmap.png',bbox_inches = 'tight')


def tsne(args, model, tokenizer, labels_feature, mode, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, eval_task, tokenizer, mode=mode)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        #logger.info("***** Running evaluation on %s.txt *****" % mode)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        crf_logits, crf_mask = [], []
        for i,batch in enumerate(eval_dataloader):
            if i>100:
                break
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                      'attention_mask':     batch[1],
                      'seq_labels':         batch[2],
                      'generate_label_ids': batch[3],
                      'step':               -1,
                      'labels_feature':     labels_feature,}
                outputs,labels_feature, com_features = model(**inputs)
                
                # logits: (bsz, seq_len, label_size)
                # here the loss is the masked loss
                tmp_eval_loss, logits = outputs[:2]
                
                dim = com_features.size()[-1]
                logits = com_features.contiguous().view(-1,dim)
                labels = inputs['seq_labels'].view(-1)
                mask = batch[1].view(-1)
                _idx = torch.where(labels!=0)[0]
                # pdb.set_trace()
                logits = torch.index_select(logits,dim=0,index=_idx).detach().cpu().numpy()
                labels = torch.index_select(labels,dim=0,index=_idx).detach().cpu().numpy()
            if preds is None:
                preds = logits
                all_labels = labels
            else:
                preds = np.append(preds, logits, axis=0)
                all_labels = np.append(all_labels, labels, axis=0)

        # all_label_features = labels_feature[1]
        # keys = list(labels_feature.keys())
        # for k in keys:
        #     if k > 1:
        #         all_label_features = torch.cat((all_label_features,labels_feature[k]), dim=0)  # 维度0-6,对应标签1-7
        # all_label_features = all_label_features.detach().cpu().numpy()
        # # pdb.set_trace()
        # logits = np.concatenate((preds,all_label_features),axis=0)
        # label_labels= np.array(range(1,8))
        # labels = np.append(all_labels,label_labels)
        logits = preds
        labels = all_labels
        t_sne_features = TSNE(n_components=2, init='pca', random_state=2022).fit_transform(logits)
        x_min, x_max = t_sne_features.min(0), t_sne_features.max(0)
        X_norm = (t_sne_features - x_min) / (x_max - x_min)
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        plt.axis('off')
        t2_x,t2_y = [],[]
        t3_x,t3_y = [],[]
        t4_x,t4_y = [],[]

        for i in range(X_norm.shape[0]):
            if labels[i] == 2:
                t2_x.append(X_norm[i][0])
                t2_y.append(X_norm[i][1])
            elif labels[i] == 3:
                t3_x.append(X_norm[i][0])
                t3_y.append(X_norm[i][1])
            elif labels[i] == 4:
                t4_x.append(X_norm[i][0])
                t4_y.append(X_norm[i][1])

        t2 = ax.scatter(t2_x,t2_y,s=3,c=plt.cm.tab10(0))
        t3 = ax.scatter(t3_x,t3_y,s=3,c=plt.cm.tab10(1))
        t4 = ax.scatter(t4_x,t4_y,s=3,c=plt.cm.tab10(2))

        ax.legend((t2,t3,t4),('T-POS', 'T-NEG', 'T-NEU'),loc=0)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # plt.tight_layout()
        # pdb.set_trace()
        # plt.scatter(X_norm[:,0], X_norm[:,1],s=10,c=labels,cmap="tab10")
        # for i in range(X_norm.shape[0]):
        #     # if i >= (X_norm.shape[0] - 7):
        #     #     plt.text(X_norm[i, 0], X_norm[i, 1], '#', color=plt.cm.Set1(labels[i]), 
        #     #             fontdict={'weight': 'bold', 'size': 20})
            # plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]), 
            #             fontdict={'weight': 'bold', 'size': 9})
        # plt.xticks([])
        # plt.yticks([])
        plt.savefig("filename.png")
        # plt.show()
        exit()


def evaluate(args, model, tokenizer, labels_feature, mode):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, eval_task, tokenizer, mode=mode)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        #logger.info("***** Running evaluation on %s.txt *****" % mode)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        crf_logits, crf_mask = [], []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                      'attention_mask':     batch[1],
                      'seq_labels':         batch[2],
                      'generate_label_ids': batch[3],
                      'step':               -1,
                      'labels_feature':     labels_feature,}
                outputs,labels_feature = model(**inputs)
                # logits: (bsz, seq_len, label_size)
                # here the loss is the masked loss
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

                crf_logits.append(logits)
                crf_mask.append(batch[1])
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['seq_labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['seq_labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        # argmax operation over the last dimension

        # viterbi decoding for CRF-based model
        crf_logits = torch.cat(crf_logits, dim=0)
        crf_mask = torch.cat(crf_mask, dim=0)
        preds = model.crf.viterbi_tags(logits=crf_logits, mask=crf_mask)
        # hotmap(crf_logits)
        result = compute_metrics_absa(preds, out_label_ids, eval_evaluate_label_ids, args.tagging_schema,args, mode)
        result['eval_loss'] = eval_loss
        results.update(result)

    return results

def load_and_cache_examples(args, task, tokenizer, mode='train'):
    processor = ABSAProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        #logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.tagging_schema)
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir, args.tagging_schema)
        elif mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.tagging_schema)
        elif mode == 'test':
            examples = processor.get_test_examples(args.data_dir, args.tagging_schema)
        else:
            raise Exception("Invalid data mode %s..." % mode)
        
        # pdb.set_trace()
        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer)
        if args.local_rank in [-1, 0]:
            #logger.info("Saving features into cached file %s", cached_features_file)
            torch.save((features), cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)  # [2741,87]
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_seqlabel_ids = torch.tensor([f.seq_label_ids for f in features], dtype=torch.long)
    all_generate_label_ids = torch.tensor([f.generate_label_ids for f in features], dtype=torch.long)
    # used in evaluation
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]

    dataset = TensorDataset(all_input_ids, all_input_mask, all_seqlabel_ids, all_generate_label_ids)
    return dataset, all_evaluate_label_ids


def main():

    args = init_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    args.n_gpu = 0
    device = torch.device('cuda',args.device)
    args.device = device

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()

    processor = ABSAProcessor()
    label_list = processor.get_labels(args.tagging_schema)
    num_labels = len(label_list)
    args.num_labels = num_labels

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # initialize the pre-trained model
    args.model_type = args.model_type.lower()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5encoder_merge(args)
    # checkpoints = os.path.join(args.output_dir, 'checkpoint-2', 'model_parameter.pkl')
    # state_dict = torch.load(checkpoints)

    # model.load_state_dict(state_dict=state_dict, strict=False)
    model.to(args.device)

    # Distributed and parallel training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Training
    if args.do_train:
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        # not using 16-bits training
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: False",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
        
        checkpoint_dir = os.path.join(args.output_dir, args.checkpoint)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        file_handler = FileHandler(filename=os.path.join(checkpoint_dir, 'training.log'), mode='w')
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        train_dataset, train_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    
    # Validation
    results = {}
    best_f1 = -999999.0
    best_checkpoint = None
    checkpoint = args.checkpoint

    test_results = {}
    train_results = {}

    model = T5encoder_merge(args)
    checkpoint_model = os.path.join(args.output_dir, checkpoint, 'model_parameter.pkl')
    state_dict = torch.load(checkpoint_model)
    label_path = os.path.join(args.output_dir, checkpoint, 'labels_feature.pth')
    labels_feature = torch.load(label_path,map_location=args.device)

    model.load_state_dict(state_dict=state_dict)
    model.to(args.device)

    if args.do_tsne:
        # tsne(args, model, tokenizer, labels_feature, mode='train', prefix='')
        tsne(args, model, tokenizer, labels_feature, mode='test', prefix='')
        
    test_result = evaluate(args, model, tokenizer, labels_feature=labels_feature, mode='test')
    print(test_result)


if __name__ == '__main__':
    main()
