import argparse
import os
import torch
import logging
from logging import FileHandler
import random
import numpy as np
import sys
from tqdm import tqdm, trange

from glue_utils_seq_gengerate import convert_examples_to_seq_features, ABSAProcessor, compute_metrics_absa
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from T5FineTuner import T5_generate
from eval_lego_4 import compute_scores
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
    parser.add_argument('--save_steps', type=int, default=200,
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
    labels_feature = None
    for _ in train_iterator:
        
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':           batch[0],
                      'attention_mask':      batch[1],
                      'seq_labels':          batch[2],
                      'generate_label_ids':  batch[3],
                      'generate_label_mask': batch[4],
                      'mode':                'train',
                      'step':                step,
                      'labels_feature':      labels_feature}
            outputs, labels_feature = model(**inputs)
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
                train_result = evaluate(args, model, tokenizer, mode='train')
                train_results = train_result['raw_scores']
                dev_result = evaluate(args, model, tokenizer, mode='dev')
                dev_results = dev_result['raw_scores']
                test_result = evaluate(args, model, tokenizer, mode='test')
                test_results = test_result['raw_scores']
                logger.info(
                    "Epoch: {} Train f1: {:.4f}, Dev_f1: {:.4f}, Test precision: {:.4f}, Test recall: {:.4f}, Test f1: {:.4f}."
                    .format(global_step, train_results['f1'], dev_results['f1'], test_results['precision'], test_results['recall'], test_results['f1'])
                    )
                if dev_results['f1'] > dev_best_f1:
                    logger.info("===================new checkpoint==================")
                    dev_best_f1 = dev_results['f1']
                    torch.save(model.state_dict(), os.path.join(output_dir, 'model_parameter.pkl'))
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


def evaluate(args, model, tokenizer, mode,):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_evaluate_label_ids, prompt_len = load_and_cache_examples(args, eval_task, tokenizer, mode=mode)

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
        outputs, targets = [], []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            
            outs = model.T5.generate(input_ids=batch[0].to(args.device), 
                                    attention_mask=batch[1].to(args.device), 
                                    max_length=128)
            dec = [tokenizer.decode(ids) for ids in outs]
            target = [tokenizer.decode(ids) for ids in batch[3]]
            outputs.extend(dec)
            targets.extend(target)
        raw_scores, all_labels, all_preds = compute_scores(outputs, targets, task='uabsa')
        results = {'raw_scores': raw_scores, 'labels': all_labels,
                   'preds': all_preds}

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
        features,prompt_len = torch.load(cached_features_file)
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
        features, prompt_len = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer)
        if args.local_rank in [-1, 0]:
            #logger.info("Saving features into cached file %s", cached_features_file)
            torch.save((features,prompt_len), cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)  # [2741,87]
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_seq_label_ids = torch.tensor([f.seq_label_ids for f in features], dtype=torch.long)

    all_generate_label_ids = torch.tensor([f.generate_label_ids for f in features], dtype=torch.long)
    all_generate_label_mask = torch.tensor([f.generate_label_mask for f in features], dtype=torch.long)

    # used in evaluation
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]

    dataset = TensorDataset(all_input_ids, all_input_mask, all_seq_label_ids, all_generate_label_ids, all_generate_label_mask)
    return dataset, all_evaluate_label_ids, prompt_len


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
    model = T5_generate(args)
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

        train_dataset, train_evaluate_label_ids,_ = load_and_cache_examples(args, args.task_name, tokenizer, mode='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    
    # Validation
    results = {}
    best_f1 = -999999.0
    best_checkpoint = None
    checkpoint = args.checkpoint

    test_results = {}
    train_results = {}

    model = T5_generate(args)
    checkpoints = os.path.join(args.output_dir, checkpoint, 'model_parameter.pkl')
    state_dict = torch.load(checkpoints)

    model.load_state_dict(state_dict=state_dict)
    model.to(args.device)

    train_result = evaluate(args, model, tokenizer, mode='train')
    dev_result = evaluate(args, model, tokenizer, mode='dev')
    test_result = evaluate(args, model, tokenizer, mode='test')
    test_false_path = os.path.join(args.output_dir, checkpoint, 'test_false.txt')
    t_false = open(test_false_path, 'w')
    for i in range(len(test_result['preds'])):
        if test_result['preds'][i] != test_result['labels'][i]:
            t_false.write("id: {}  pred labels:{}  gold labels:{}. \n".format(i, str(test_result['preds'][i]), str(test_result['labels'][i])))
    # print(test_result)
    train_pred_path = os.path.join(args.data_dir, "train_pred_generate.txt")
    dev_pred_path = os.path.join(args.data_dir, "dev_pred_generate.txt")
    test_pred_path = os.path.join(args.data_dir, "test_pred_generate.txt")
    f_train = open(train_pred_path, 'w+')
    f_dev = open(dev_pred_path,'w+')
    f_test = open(test_pred_path, 'w+')
    for i in range(len(train_result['preds'])):
        f_train.write(str(train_result['preds'][i]) + '\n')
    for i in range(len(dev_result['preds'])):
        f_dev.write(str(dev_result['preds'][i]) + '\n')
    for i in range(len(test_result['preds'])):
        f_test.write(str(test_result['preds'][i]) + '\n')
    f_train.close()
    f_dev.close()
    f_test.close()


if __name__ == '__main__':
    main()
