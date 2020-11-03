import pickle
import re
import os

import random
import numpy as np
import torch
from random import shuffle
import argparse
import pickle

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.BERT import *

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from util.optimization import BERTAdam
from util.processor import *

from util.tokenization import *

from util.evaluation import *

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# prepare dataloaders
processors = {
    "SST5" : SST5_Processor
}

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_len):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.seq_len = seq_len

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_len = len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = int(example.label)
        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        seq_len=seq_len))
    
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def load_model_setups(vocab_file, bert_config_file,
                        init_checkpoint,
                        label_list, 
                        num_train_steps,
                        do_lower_case=True, 
                        learning_rate=2e-5,
                        warmup_proportion=0.1):
    logger.info("model = BERT")
    if bert_config_file is not None:
        bert_config = BertConfig.from_json_file(bert_config_file)
    else:
        # default?
        bert_config = BertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02
        )
    logger.info("*** Model Config ***")
    logger.info(bert_config.to_json_string())
    tokenizer = FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case, pretrain=False)
    # overwrite the vocab size to be exact. this also save space incase
    # vocab size is shrinked.
    bert_config.vocab_size = len(tokenizer.vocab)
    # model and optimizer
    model = BertForSequenceClassification(bert_config, len(label_list))
    if init_checkpoint is None:
        err_msg = "Error: model have to be based on a pretrained model"
        logger.warning(err_msg)
        raise Exception(err_msg)
    
    # checkpoint should be used only for generated model during training
    if "checkpoint" in init_checkpoint:
        # we need to add handling logic specially for parallel gpu trainign
        state_dict = torch.load(init_checkpoint, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v
            else:
                new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
    else:
        logger.info("retraining with saved model.")
        model.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() 
            if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() 
            if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    optimizer = BERTAdam(optimizer_parameters,
                        lr=learning_rate,
                        warmup=warmup_proportion,
                        t_total=num_train_steps)
    return model, tokenizer, optimizer

def step_train(train_dataloader, test_dataloader, model, optimizer, 
               device, n_gpu, evaluate_interval, global_step, 
               output_log_file, epoch, global_best_acc, args):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    pbar = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(pbar):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # truncate to save space and computing resource
        input_ids, input_mask, segment_ids, label_ids, seq_lens = batch
        max_seq_lens = max(seq_lens)[0]
        input_ids = input_ids[:,:max_seq_lens]
        input_mask = input_mask[:,:max_seq_lens]
        segment_ids = segment_ids[:,:max_seq_lens]

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        seq_lens = seq_lens.to(device)

        loss, _ = \
            model(input_ids, segment_ids, input_mask, seq_lens,
                            device=device, labels=label_ids)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()    # We have accumulated enought gradients
            model.zero_grad()
            global_step += 1
        pbar.set_postfix({'train_loss': loss.tolist()})

        if global_step % 500 == 0:
            logger.info("***** Evaluation Interval Hit *****")
            global_best_acc = evaluate(test_dataloader, model, device, n_gpu, nb_tr_steps, tr_loss, epoch, 
                                       global_step, output_log_file, global_best_acc, args)

    return global_step, global_best_acc

def evaluate_fast(test_dataloader, model, device, n_gpu, args):
    """
    evaluate only and not recording anything
    """
    # eval_test
    model.eval()
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    # we don't need gradient in this case.
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids, seq_lens in test_dataloader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # truncate to save space and computing resource
            max_seq_lens = max(seq_lens)[0]
            input_ids = input_ids[:,:max_seq_lens]
            input_mask = input_mask[:,:max_seq_lens]
            segment_ids = segment_ids[:,:max_seq_lens]

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            seq_lens = seq_lens.to(device)

            # intentially with gradient
            tmp_test_loss, logits = \
                model(input_ids, segment_ids, input_mask, seq_lens,
                        device=device, labels=label_ids)

            logits = F.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            tmp_test_accuracy=np.sum(outputs == label_ids)

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_accuracy

            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples

    result = collections.OrderedDict()

    result = {'test_loss': test_loss,
                'test_accuracy': test_accuracy}
    for key in result.keys():
        logger.info("  %s = %s\n", key, str(result[key]))

    return test_accuracy

def evaluate(test_dataloader, model, device, n_gpu, nb_tr_steps, tr_loss, epoch, 
             global_step, output_log_file, global_best_acc, args):
    # eval_test
    if args.eval_test:
        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        # we don't need gradient in this case.
        with torch.no_grad():
            for input_ids, input_mask, segment_ids, label_ids, seq_lens in test_dataloader:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # truncate to save space and computing resource
                max_seq_lens = max(seq_lens)[0]
                input_ids = input_ids[:,:max_seq_lens]
                input_mask = input_mask[:,:max_seq_lens]
                segment_ids = segment_ids[:,:max_seq_lens]

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                seq_lens = seq_lens.to(device)

                # intentially with gradient
                tmp_test_loss, logits = \
                    model(input_ids, segment_ids, input_mask, seq_lens,
                            device=device, labels=label_ids)

                logits = F.softmax(logits, dim=-1)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                outputs = np.argmax(logits, axis=1)
                tmp_test_accuracy=np.sum(outputs == label_ids)

                test_loss += tmp_test_loss.mean().item()
                test_accuracy += tmp_test_accuracy

                nb_test_examples += input_ids.size(0)
                nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples

    result = collections.OrderedDict()
    # handling corner case for a checkpoint start
    if nb_tr_steps == 0:
        loss_tr = 0.0
    else:
        loss_tr = tr_loss/nb_tr_steps
    if args.eval_test:
        result = {'epoch': epoch,
                'global_step': global_step,
                'loss': loss_tr,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy}
    else:
        result = {'epoch': epoch,
                'global_step': global_step,
                'loss': loss_tr}

    logger.info("***** Eval results *****")
    with open(output_log_file, "a+") as writer:
        for key in result.keys():
            logger.info("  %s = %s\n", key, str(result[key]))
            writer.write("%s\t" % (str(result[key])))
        writer.write("\n")

    # save for each time point
    if args.eval_test and args.output_dir:
        torch.save(model.state_dict(), args.output_dir + "checkpoint_" + str(global_step) + ".bin")
        if test_accuracy > global_best_acc:
            torch.save(model.state_dict(), args.output_dir + "best_checkpoint.bin")
            global_best_acc = test_accuracy

    return global_best_acc

def data_and_model_loader(device, n_gpu, args):
    processor = processors[args.task_name]()
    label_list = processor.get_labels()

    # training loop
    train_examples = None
    num_train_steps = None
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size * args.num_train_epochs)

    model, tokenizer, optimizer = \
        load_model_setups(args.vocab_file, args.bert_config_file,
                            args.init_checkpoint, label_list, num_train_steps)

    # training set
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length,
        tokenizer)
    logger.info("***** Running training for model *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_seq_len = torch.tensor([[f.seq_len] for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_label_ids, all_seq_len)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=10)

    # test set stays the same
    test_dataloader = None
    if args.eval_test:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length,
            tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_seq_len = torch.tensor([[f.seq_len] for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                  all_label_ids, all_seq_len)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)    

    return model, optimizer, train_dataloader, test_dataloader

def system_setups(args):
    # system related setups
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    # not preloading
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=",output_log_file)

    # initialize output files
    with open(output_log_file, "w") as writer:
        if args.eval_test:
            writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
        else:
            writer.write("epoch\tglobal_step\tloss\n")

    return device, n_gpu, output_log_file