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

def get_inputivation(name):
    def hook(model, input, output):
        func_inputs[name] = [_in for _in in input]
    return hook

def get_activation(name):
    def hook(model, input, output):
        func_activations[name] = output
    return hook

def get_activation_multi(name):
    def hook(model, input, output):
        func_activations[name] = [_out for _out in output]
    return hook

# TODO: make this init as a part of the model init
def init_hooks_lrp(model):
    """
    Initialize all the hooks required for full lrp for BERT model.
    """
    # in order to backout all the lrp through layers
    # you need to register hooks here.

    model.classifier.register_forward_hook(
        get_inputivation('model.classifier'))
    model.classifier.register_forward_hook(
        get_activation('model.classifier'))

    model.word_embeddings.register_forward_hook(
        get_activation('model.word_embeddings'))

    layer_module_index = 0
    for module_layer in model.attendedEncoder.layer_stack:
        
        # self-attention layer
        layer_name_slf_attn = \
            'model.attendedEncoder.layer_stack.' + \
            str(layer_module_index) + \
            '.slf_attn'
        module_layer.slf_attn.register_forward_hook(
            get_inputivation(layer_name_slf_attn))
        module_layer.slf_attn.register_forward_hook(
            get_activation_multi(layer_name_slf_attn))
        
        layer_module_index += 1
    
def pad_shift(x, shift, padv=0.0):
    """Shift 3D tensor forwards in time with padding."""
    if shift > 0:
        padding = torch.ones(x.size(0), shift, x.size(2)).to(x.device) * padv
        return torch.cat((padding, x[:, :-shift, :]), dim=1)
    elif shift < 0:
        padding = torch.ones(x.size(0), -shift, x.size(2)).to(x.device) * padv
        return torch.cat((x[:, -shift:, :], padding), dim=1)
    else:
        return x

def convolve(x, attn):
    """Convolve 3D tensor (x) with local attention weights (attn)."""
    stacked = torch.stack([pad_shift(x, i) for
                           i in range(attn.shape[2])], dim=-1)
    return torch.sum(attn.unsqueeze(2) * stacked, dim=-1)

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask=None):
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.temperature
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        return context_layer, attention_probs

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)

        # ops on v to save all the context for backing out
        v_ma_first_pre = v.clone()
        v_ma_first_post = self.w_vs(v)
        v_ma_first_post_ret = v_ma_first_post.clone()
        v = v_ma_first_post.view(sz_b, len_v, n_head, d_v)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        v_attended, attn = self.attention(q, k, v, attention_mask=mask)

        v_attended = v_attended.transpose(1, 2).contiguous()

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = v_attended.view(sz_b, len_q, -1)

        q = self.fc(q)
        q = self.dropout(q)
        q += residual

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.layer_norm(x)
        x = F.relu(self.w_1(x))

        x = self.dropout(self.w_2(x))
        
        x += residual

        return x
    
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn
    
class attendedEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, masks):

        # -- Forward
        enc_output = self.dropout(inputs)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=masks)

        enc_output = self.layer_norm(enc_output)

        return enc_output

class TransformerSequenceClassification(nn.Module):

    def __init__(self, vocab_size, n_labels=3, embeddings=None, init_lrp=False):
        super(TransformerSequenceClassification, self).__init__()

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
        self.attendedEncoder = attendedEncoder(self.att_n_layer,
                                               self.att_n_header,
                                               att_d_k,
                                               self.att_d_v,
                                               att_d_model,
                                               att_d_inner)

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
        attended_out = \
            self.attendedEncoder(inputs,
                                 extended_attention_mask)

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
        