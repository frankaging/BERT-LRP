from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

import collections
from functools import partial

# access global vars here
global func_inputs
global func_activations
func_inputs = collections.defaultdict(list)
func_activations = collections.defaultdict(list)

class LSTMSequenceClassification(nn.Module):

    def __init__(self, vocab_size, n_labels=3, embeddings=None, init_lrp=False):
        super(LSTMSequenceClassification, self).__init__()

        # config
        self.att_n_layer = 6
        self.att_n_header = 8
        self.encoder_in = 300
        self.encoder_out = 300
        att_d_k = 64
        self.att_d_v = 64
        att_d_model = 300
        att_d_inner = 64
        
        # embedding layer
        if embeddings != None:
            self.word_embeddings = nn.Embedding(vocab_size, self.encoder_in)
            self.word_embeddings.weight.data.copy_(embeddings.weight)
            if not init_lrp:
                self.word_embeddings.weight.requires_grad = False
        else:
            self.word_embeddings = nn.Embedding(vocab_size, self.encoder_in)
        
        # transformer
        self.attendedEncoder = nn.LSTM(self.encoder_in,  self.encoder_out, batch_first=True)

        # context vector
        attn_dropout = 0.1
        self.dropout = nn.Dropout(attn_dropout)
        self.encoder_gate = nn.Sequential(nn.Linear(self.encoder_out, 128),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(64, 1))

        # final output layers
        final_out = self.encoder_out
        h_out = 64
        out_dropout = 0.3
        self.out_fc1 = nn.Linear(final_out, h_out)
        self.classifier = nn.Linear(h_out, n_labels)
        
        if init_lrp:
            print("init_lrp = True")
            init_hooks_lrp(self)

    def forward(self, input_ids, input_mask, seq_lens, labels=None):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # embeddings
        inputs = self.word_embeddings(input_ids)
        
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, from_seq_length]
        # So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # transformer encoder
        attended_out, (_, _) = self.attendedEncoder(inputs)

        mask_bool = input_mask.bool()
        mask_bool = mask_bool.unsqueeze(dim=-1)
        
        # context layer
        attn = self.encoder_gate(attended_out)
        attn = attn.masked_fill(mask_bool == 0, -1e9)
        
        attn = F.softmax(attn, dim=1)
        # attened embeddings
        hs_attend = \
            torch.matmul(attn.permute(0,2,1), attended_out).squeeze(dim=1)
        # final output blocks
        hs_attend = self.out_fc1(hs_attend)
        hs_attend = F.relu(hs_attend)
        logits = self.classifier(hs_attend)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, attn
        else:
            return logits
        
    def backward_gradient(self, sensitivity_grads):
        classifier_out = func_activations['model.classifier']
        embedding_output = func_activations['model.word_embeddings']
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output, 
                                                grad_outputs=sensitivity_grads, 
                                                retain_graph=True)[0]
        return sensitivity_grads
    
    def backward_lat(self, input_ids, ctx_attn):
        
        enc_slf_attn_list = []
        for layer_indx in range(self.att_n_layer):
            # self-attention layer
            layer_name_slf_attn = \
                'model.attendedEncoder.layer_stack.' + \
                str(layer_indx) + \
                '.slf_attn'
            enc_slf_attn = func_activations[layer_name_slf_attn][1]
            enc_slf_attn_list += [enc_slf_attn]
        tf_attns = torch.stack(enc_slf_attn_list, dim=0).permute(2,0,1,3,4)

        raw_attns = []
        for h in range(tf_attns.shape[0]):
            tf_attn = tf_attns[h]
            pre_attn = ctx_attn.clone().permute(0, 2, 1)
            for i in reversed(range(self.att_n_layer)):
                curr_tf_attn = torch.matmul(pre_attn, tf_attn[i])
                pre_attn = curr_tf_attn
            raw_attns.append(pre_attn.permute(0,2,1))

        raw_attns = torch.stack(raw_attns, dim=0).sum(dim=0)
        return raw_attns
        