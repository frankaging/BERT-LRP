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
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.BiLSTM import *
from model.BERT import *

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from util.optimization import BERTAdam
from util.processor import (IMDb_Processor,
                            SemEval_Processor,
                            SST5_Processor,
                            SST2_Processor,
                            Yelp5_Processor,
                            Yelp2_Processor)

from util.tokenization import *

from util.evaluation import *

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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


def getModelOptimizerTokenizer(model_type, vocab_file, embed_file=None, 
                               bert_config_file=None, init_checkpoint=None,
                               label_list=None,
                               do_lower_case=True,
                               num_train_steps=None,
                               learning_rate=None,
                               base_learning_rate=None,
                               warmup_proportion=None):
    if embed_file is not None:
        # in case pretrain embeddings
        embeddings = pickle.load(open(embed_file, 'rb'))

    if model_type == "BiLSTM":
        logger.info("model = BiLSTM")
        tokenizer = WordLevelTokenizer(vocab_file=vocab_file)
        model = BiLSTM(pretrain_embeddings=embeddings, freeze=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # if pretrain, we will load here
        if init_checkpoint is not None:
            logger.info("retraining with saved model.")
            checkpoint = torch.load(init_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint)
    elif model_type == "BERTPretrain":
        logger.info("model = BERTPretrain")
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
        tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case, pretrain=False)
        # overwrite the vocab size to be exact. this also save space incase
        # vocab size is shrinked.
        bert_config.vocab_size = len(tokenizer.vocab)
        # model and optimizer
        model = BertForSequenceClassification(bert_config, len(label_list), 
                                              init_lrp=True)

        if init_checkpoint is not None:
            if "checkpoint" in init_checkpoint:
                # we need to add handling logic specially for parallel gpu trainign
                state_dict = torch.load(init_checkpoint, map_location='cpu')
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove 'module.' of dataparallel
                    new_state_dict[name]=v
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

    else:
        logger.info("***** Not Support Model Type *****")
    return model, optimizer, tokenizer

def LRP(args):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.bert_config_file is not None:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        if args.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    # prepare dataloaders
    processors = {
        "IMDb":IMDb_Processor,
        "SemEval":SemEval_Processor,
        "SST5":SST5_Processor,
        "SST2":SST2_Processor,
        "Yelp5":Yelp5_Processor,
        "Yelp2":Yelp2_Processor
    }

    processor = processors[args.task_name]()
    label_list = processor.get_labels()

    # model and tokenizer are all we care
    model, _, tokenizer = \
        getModelOptimizerTokenizer(model_type=args.model_type,
                                   vocab_file=args.vocab_file,
                                   embed_file=args.embed_file,
                                   bert_config_file=args.bert_config_file,
                                   init_checkpoint=args.init_checkpoint,
                                   label_list=label_list,
                                   do_lower_case=True,
                                   # these parameters, we are not going to use it
                                   num_train_steps=1,
                                   learning_rate=1e-4,
                                   base_learning_rate=1e-4,
                                   warmup_proportion=0.1)

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

    logger.info("***** Running lrp on test examples *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    # we recommand to run lrp in 1 cpu
    elif n_gpu > 1:
        logger.info("***** WARNING: Please run LRP on a single GPU *****")
        model = torch.nn.DataParallel(model)

    model.to(device)

    if "checkpoint" in args.init_checkpoint:
        logger.info("loading previous checkpoint model not the pretrain BERT-base...")

    lrp_scores = []
    inputs_ids = []
    seqs_lens = []

    model.eval() # this line will deactivate dropouts
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    # we don't need gradient in this case.
    for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        input_ids, input_mask, segment_ids, label_ids, seq_lens = batch
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

        # for lrp
        LRP_class = 4
        Rout_mask = torch.zeros((input_ids.shape[0], len(label_list))).to(device)
        Rout_mask[:, LRP_class] = 1.0
        logits = func_activations['model.classifier']
        relevance_score = logits*Rout_mask
        lrp_score = model.backward_lrp(relevance_score).sum(dim=-1).cpu().data
        input_ids = input_ids.cpu().data
        seq_lens = seq_lens.cpu().data
        lrp_scores.append(lrp_score)
        inputs_ids.append(input_ids)
        seqs_lens.append(seq_lens)

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
    logger.info("***** Eval results *****")
    for key in result.keys():
        logger.info("  %s = %s\n", key, str(result[key]))

    # also save lrp results into a pickle file
    lrp_state_dict = dict()
    lrp_state_dict["lrp_scores"] = lrp_scores
    lrp_state_dict["inputs_ids"] = inputs_ids
    lrp_state_dict["seqs_lens"] = seqs_lens
    logger.info("***** Saving LRP to disk *****")
    torch.save(lrp_state_dict, os.path.join(args.output_dir + 'lrp_state.pt'))
    logger.info("***** Finish LRP *****")

def router(args):
    LRP(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["IMDb", "SemEval",
                                 "SST2", "SST5", 
                                 "Yelp2", "Yelp5"],
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument('--model_type', 
                        type=str,
                        default=None,
                        required=True,
                        help='type of model to evaluation with lrp')  
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="Initial checkpoint (usually from a pre-trained model).")       
    ## Other parameters
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--embed_file",
                        default=None,
                        type=str,
                        help="The embedding file that the model was trained on.")          
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")             
    args = parser.parse_args()
    router(args)