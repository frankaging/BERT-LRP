# coding=utf-8

# Reference: https://github.com/huggingface/pytorch-pretrained-BERT

"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function

import copy
import json
import math

import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import collections
from functools import partial

from util.lrp import *

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
    model.bert.pooler.dense.register_forward_hook(
        get_inputivation('model.bert.pooler.dense'))
    model.bert.pooler.dense.register_forward_hook(
        get_activation('model.bert.pooler.dense'))
    model.bert.pooler.register_forward_hook(
        get_inputivation('model.bert.pooler'))
    model.bert.pooler.register_forward_hook(
        get_activation('model.bert.pooler'))

    model.bert.embeddings.word_embeddings.register_forward_hook(
        get_activation('model.bert.embeddings.word_embeddings'))
    model.bert.embeddings.register_forward_hook(
        get_activation('model.bert.embeddings'))

    layer_module_index = 0
    for module_layer in model.bert.encoder.layer:
        
        ## Encoder Output Layer
        layer_name_output_layernorm = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.output.LayerNorm'
        module_layer.output.LayerNorm.register_forward_hook(
            get_inputivation(layer_name_output_layernorm))

        layer_name_dense = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.output.dense'
        module_layer.output.dense.register_forward_hook(
            get_inputivation(layer_name_dense))
        module_layer.output.dense.register_forward_hook(
            get_activation(layer_name_dense))

        layer_name_output = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.output'
        module_layer.output.register_forward_hook(
            get_inputivation(layer_name_output))
        module_layer.output.register_forward_hook(
            get_activation(layer_name_output))
        
        ## Encoder Intermediate Layer
        layer_name_inter = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.intermediate.dense'
        module_layer.intermediate.dense.register_forward_hook(
            get_inputivation(layer_name_inter))
        module_layer.intermediate.dense.register_forward_hook(
            get_activation(layer_name_inter))

        layer_name_attn_layernorm = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.output.LayerNorm'
        module_layer.attention.output.LayerNorm.register_forward_hook(
            get_inputivation(layer_name_attn_layernorm))
        
        layer_name_attn = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.output.dense'
        module_layer.attention.output.dense.register_forward_hook(
            get_inputivation(layer_name_attn))
        module_layer.attention.output.dense.register_forward_hook(
            get_activation(layer_name_attn))

        layer_name_attn_output = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.output'
        module_layer.attention.output.register_forward_hook(
            get_inputivation(layer_name_attn_output))
        module_layer.attention.output.register_forward_hook(
            get_activation(layer_name_attn_output))
        
        layer_name_self = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self'
        module_layer.attention.self.register_forward_hook(
            get_inputivation(layer_name_self))
        module_layer.attention.self.register_forward_hook(
            get_activation_multi(layer_name_self))

        layer_name_value = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.value'
        module_layer.attention.self.value.register_forward_hook(
            get_inputivation(layer_name_value))
        module_layer.attention.self.value.register_forward_hook(
            get_activation(layer_name_value))

        layer_name_query = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.query'
        module_layer.attention.self.query.register_forward_hook(
            get_inputivation(layer_name_query))
        module_layer.attention.self.query.register_forward_hook(
            get_activation(layer_name_query))

        layer_name_key = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.key'
        module_layer.attention.self.key.register_forward_hook(
            get_inputivation(layer_name_key))
        module_layer.attention.self.key.register_forward_hook(
            get_activation(layer_name_key))
        
        layer_module_index += 1

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size=32000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02,
                full_pooler=False): # this is for transformer-like BERT
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.full_pooler = full_pooler

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def backward_lrp(self, relevance_score):
        # we use the whole embedding as its units
        return relevance_score

class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_context(self, x):
        new_x_shape = x.size()[:2] + \
            (self.num_attention_heads, self.attention_head_size,)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def transpose_for_value(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:2] + (self.all_head_size,)
        return x.view(*new_x_shape)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs
    
    def attention_core(self, query_layer, key_layer, value_layer, attention_mask):
        """
        This is the core self-attention layer.
        """
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        return attention_probs
    
    def jacobian(self, tensor_out, tensor_in, debug=False):
        """
        This is super slow. You can simply write out the full
        jacboian by hand, and it would be so much faster.
        PyTorch team is working on a fastor impl which is still
        in progress.
        """
        import time
        start = time.time()
        jacobian_full = []
        for i in range(tensor_out.shape[2]):
            jac_mask = torch.zeros_like(tensor_out)
            jac_mask[:,:,i] = 1.
            jacobian_partial = torch.autograd.grad(tensor_out, tensor_in,
                                                   grad_outputs=jac_mask,
                                                   retain_graph=True)[0]
            jacobian_full.append(jacobian_partial)
        jacobian_full = torch.stack(jacobian_full, dim=2)
        end = time.time()
        if debug:
            print(jacobian_full.shape)
            print("Time Elapse for 1 Jacobian Full: ", end - start)
        return jacobian_full

    def _attn_head_jacobian(self, q, k, v, attn_mask):
        """ 
        same as jacobian above, but faster 
        referene code: 
        https://github.com/lena-voita/the-story-of-heads/blob/master/lib/layers/attn_lrp.py
        """
        # input shapes: (q, k, v) - [batch_size, n_q or n_kv, dim per head]
        # attn_head_mask: [batch_size, n_q, n_kv]
        assert len(q.shape) == 3 and len(attn_mask.shape) == 3
        
        ATTN_BIAS_VALUE = -1e9
        key_depth_per_head = float(q.shape[-1])
        q = q / (key_depth_per_head ** 0.5)

        attn_bias = ATTN_BIAS_VALUE * (1 - attn_mask)
        logits = torch.matmul(q, k.permute(0,2,1)) + attn_bias
        weights = nn.Softmax(dim=-1)(logits)  # [batch_size, n_q, n_kv]
        out = torch.matmul(weights, v)  # [batch_size, n_q, dim/n_heads]

        batch_size, n_kv, dim_per_head = v.shape[0], v.shape[1], v.shape[2]

        diag_flat_weights = torch.einsum('ij,jqk->iqjk', 
                                         torch.eye(weights.shape[0]), weights)  # [b, n_q, b, n_kv]
        flat_jac_v = diag_flat_weights[:, :, None, :, :, None] * \
                        torch.eye(dim_per_head)[None, None, :, None, None, :]
        # ^-- shape: [batch_size, n_q, dim/h, batch_size, n_kv, dim/h]
        # torch.Size([1, 48, 64, 1, 48, 64])

        # ... just to get around this torch.tile(v[:, None], [1, out.shape[1], 1, 1])
        jac_out_wrt_weights = torch.cat(out.shape[1]*[v[:, None]], dim=1) 
        jac_out_wrt_weights = jac_out_wrt_weights.permute([0, 1, 3, 2])
        # ^-- [batch_size, n_q, (dim), (n_kv)]
        
        softmax_jac = (weights[..., None] * torch.eye(weights.shape[-1])
                       - weights[..., None, :] * weights[..., :, None])  # <-- [batch_size, n_q, n_kv, n_kv]
        jac_out_wrt_logits = jac_out_wrt_weights @ softmax_jac  # [batch_size, n_q, (dim), (n_kv)]

        jac_out_wrt_k = jac_out_wrt_logits[..., None] * q[:, :, None, None, :]  # [b, (n_q, dim), (n_kv, dim)]
        
        # product axes:                    b  q  d  kv   d       b  q      d    kv d
        jac_out_wrt_q = jac_out_wrt_logits[:, :, :, :, None] * k[:, None, None, :, :]
        jac_out_wrt_q = jac_out_wrt_q.sum(dim=3, keepdim=True)
        jac_out_wrt_q = jac_out_wrt_q / float(key_depth_per_head) ** 0.5
        jac_out_wrt_q = jac_out_wrt_q * torch.eye(jac_out_wrt_q.shape[1])[None, :, None, :, None]

        flat_jac_k = jac_out_wrt_k[..., None, :, :] * torch.eye(q.shape[0])[:, None, None, :, None, None]
        flat_jac_q = jac_out_wrt_q[..., None, :, :] * torch.eye(q.shape[0])[:, None, None, :, None, None]
        # final shape of flat_jac_{q, k}: [(batch_size, n_q, dim), (batch_size, n_kv, dim)]

        return flat_jac_q, flat_jac_k, flat_jac_v
    
    def backward_lrp(self, relevance_score, layer_module_index, lrp_detour="quick"):
        """
        This is the lrp explicitily considering the attention layer.
        """

        layer_name_value = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.value'
        layer_name_query = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.query'
        layer_name_key = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.key'
        value_in = func_inputs[layer_name_value][0]
        value_out = func_activations[layer_name_value]
        query_in = func_inputs[layer_name_query][0]
        query_out = func_activations[layer_name_query]
        key_in = func_inputs[layer_name_key][0]
        key_out = func_activations[layer_name_key]
        layer_name_self = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self'
        context_layer = func_activations[layer_name_self][0]
        attention_mask = func_inputs[layer_name_self][1]
        if lrp_detour == "quick":
            # Instead of jacobian, we may estimate this using a linear layer
            # This turns out to be a good estimate in general.
            relevance_query = \
                torch.autograd.grad(context_layer, query_out, 
                                    grad_outputs=relevance_score, 
                                    retain_graph=True)[0]
            relevance_key = \
                torch.autograd.grad(context_layer, key_out, 
                                    grad_outputs=relevance_score, 
                                    retain_graph=True)[0]
            relevance_value = \
                torch.autograd.grad(context_layer, value_out, 
                                    grad_outputs=relevance_score, 
                                    retain_graph=True)[0]

            relevance_query = backprop_lrp_fc(self.query.weight,
                                              self.query.bias,
                                              query_in,
                                              relevance_query)
            relevance_key = backprop_lrp_fc(self.key.weight,
                                              self.key.bias,
                                              key_in,
                                              relevance_key)
            relevance_value = backprop_lrp_fc(self.value.weight,
                                              self.value.bias,
                                              value_in,
                                              relevance_value)
            relevance_score = relevance_query + relevance_key + relevance_value
        elif lrp_detour == "jacobian":
            print("Full Jacobian can be very slow. Consider our validated quick method.")
            query_out_head = self.transpose_for_scores(query_out)
            key_out_head = self.transpose_for_scores(key_out)
            value_out_head = self.transpose_for_scores(value_out)
            relevance_score = self.transpose_for_context(relevance_score) # [b, n_h, seq_l, h_dim]

            b_n, n_h, seq_l, h_dim = query_out_head.shape[0], query_out_head.shape[1], query_out_head.shape[2], query_out_head.shape[3]
            query_out_head_flat = query_out_head.reshape([-1, seq_l, h_dim])
            key_out_head_flat = key_out_head.reshape([-1, seq_l, h_dim])
            value_out_head_flat = value_out_head.reshape([-1, seq_l, h_dim])
            relevance_score_flat = relevance_score.reshape([-1, seq_l, h_dim])
            attention_mask_flat = torch.cat(n_h*[attention_mask], dim=1).reshape([-1, 1, seq_l])
            
            # flatten them to save memory
            flat_relevence_qs = []
            flat_relevence_ks = []
            flat_relevence_vs = []
            for i in range(relevance_score_flat.shape[0]):
                flat_jac_q, flat_jac_k, flat_jac_v = \
                    self._attn_head_jacobian(query_out_head_flat[i, None],
                                             key_out_head_flat[i, None],
                                             value_out_head_flat[i, None],
                                             attention_mask_flat[i, None])
                output_flat = self.attention_core(query_out_head_flat[i, None], 
                                               key_out_head_flat[i, None], 
                                               value_out_head_flat[i, None], 
                                               attention_mask_flat[i, None])
                flat_relevence_q, flat_relevence_k, flat_relevence_v = \
                    backprop_lrp_jacobian((flat_jac_q, flat_jac_k, flat_jac_v), 
                                          output_flat, 
                                          relevance_score_flat[i, None], 
                                          (query_out_head_flat[i, None], 
                                          key_out_head_flat[i, None], 
                                          value_out_head_flat[i, None]))
                flat_relevence_qs.append(flat_relevence_q)
                flat_relevence_ks.append(flat_relevence_k)
                flat_relevence_vs.append(flat_relevence_v)
            flat_relevence_qs = torch.stack(flat_relevence_qs, dim=0)
            flat_relevence_ks = torch.stack(flat_relevence_ks, dim=0)
            flat_relevence_vs = torch.stack(flat_relevence_vs, dim=0)
            relevance_query = flat_relevence_qs.reshape(b_n, n_h, seq_l, h_dim).contiguous().permute(0,2,1,3).reshape(b_n, seq_l, -1).contiguous()
            relevance_key = flat_relevence_ks.reshape(b_n, n_h, seq_l, h_dim).contiguous().permute(0,2,1,3).reshape(b_n, seq_l, -1).contiguous()
            relevance_value = flat_relevence_vs.reshape(b_n, n_h, seq_l, h_dim).contiguous().permute(0,2,1,3).reshape(b_n, seq_l, -1).contiguous()
            
            # linear layers and we are done!
            relevance_query = backprop_lrp_fc(self.query.weight,
                                              self.query.bias,
                                              query_in,
                                              relevance_query)
            relevance_key = backprop_lrp_fc(self.key.weight,
                                              self.key.bias,
                                              key_in,
                                              relevance_key)
            relevance_value = backprop_lrp_fc(self.value.weight,
                                              self.value.bias,
                                              value_in,
                                              relevance_value)
            relevance_score = relevance_query + relevance_key + relevance_value
        return relevance_score

class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def backward_lrp(self, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'model.bert.encoder.' + str(layer_module_index) + \
                        '.attention.output'
        output_in_input = func_inputs[layer_name][1]
        output_out = func_activations[layer_name]
        relevance_score_residual = \
            torch.autograd.grad(output_out, output_in_input, 
                                grad_outputs=relevance_score, 
                                retain_graph=True)[0]
        # main connection
        layer_name_dense = 'model.bert.encoder.' + str(layer_module_index) + \
                            '.attention.output.dense'
        dense_out = func_activations[layer_name_dense]
        relevance_score = \
            torch.autograd.grad(output_out, dense_out, 
                                grad_outputs=relevance_score, 
                                retain_graph=True)[0]
        dense_in = func_inputs[layer_name_dense][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        return relevance_score, relevance_score_residual

class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs

    def backward_lrp(self, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = \
            self.output.backward_lrp(relevance_score, layer_module_index)
        relevance_score = self.self.backward_lrp(relevance_score, layer_module_index)
        # merge
        relevance_score = relevance_score + relevance_score_residual
        return relevance_score

class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def backward_lrp(self, relevance_score, layer_module_index):
        layer_name = 'model.bert.encoder.' + str(layer_module_index) + \
                        '.intermediate.dense'
        dense_in = func_inputs[layer_name][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        return relevance_score

class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def backward_lrp(self, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'model.bert.encoder.' + str(layer_module_index) + \
                        '.output'
        output_in_input = func_inputs[layer_name][1]
        output_out = func_activations[layer_name]
        relevance_score_residual = \
            torch.autograd.grad(output_out, output_in_input, 
                                grad_outputs=relevance_score, 
                                retain_graph=True)[0]
        # main connection
        layer_name_dense = 'model.bert.encoder.' + str(layer_module_index) + \
                            '.output.dense'
        dense_out = func_activations[layer_name_dense]
        relevance_score = \
            torch.autograd.grad(output_out, dense_out, 
                                grad_outputs=relevance_score, 
                                retain_graph=True)[0]
        dense_in = func_inputs[layer_name_dense][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        return relevance_score, relevance_score_residual

class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

    def backward_lrp(self, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = self.output.backward_lrp(relevance_score, layer_module_index)
        relevance_score = self.intermediate.backward_lrp(relevance_score, layer_module_index)
        # merge
        relevance_score += relevance_score_residual
        relevance_score = self.attention.backward_lrp(relevance_score, layer_module_index)
        return relevance_score

class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    
        self.num_hidden_layers = config.num_hidden_layers

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_attention_scores = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            all_encoder_attention_scores.append(attention_probs.data)
        return all_encoder_layers, all_encoder_attention_scores

    def backward_lrp(self, relevance_score):
        # backout layer by layer from last to the first
        layer_module_index = self.num_hidden_layers - 1
        for layer_module in reversed(self.layer):
            relevance_score = layer_module.backward_lrp(relevance_score, layer_module_index)
            layer_module_index -= 1
    
        # These helps you to understand how each layer
        # shift the relevance scores if any.
        # instead of go through every layer, interrupt
        # layer_name_self = 'model.bert.encoder.' + str(0) + \
        #                         '.attention.self'
        # self_attn_in = func_inputs[layer_name_self][0]
        # embedding_output = func_activations['model.bert.embeddings']
        # relevance_score = torch.autograd.grad(self_attn_in, embedding_output, 
        #                                       grad_outputs=relevance_score, 
        #                                       retain_graph=True)[0]
        return relevance_score

class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, optional_attn_mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        #return first_token_tensor
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def backward_lrp(self, relevance_score):
        dense_in = func_inputs['model.bert.pooler.dense'][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)        
        # we need to scatter this to all hidden states, but only first
        # one matters!
        pooler_in = func_inputs['model.bert.pooler'][0]
        relevance_score_all = torch.zeros_like(pooler_in)
        relevance_score_all[:, 0] = relevance_score
        return relevance_score_all

class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, from_seq_length]
        # So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers, all_encoder_attention_scores = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output, optional_attn_mask=attention_mask)
        return all_encoder_layers, pooled_output, all_encoder_attention_scores, embedding_output

    def backward_lrp(self, relevance_score):
        relevance_score = self.pooler.backward_lrp(relevance_score)
        relevance_score = self.encoder.backward_lrp(relevance_score)
        return relevance_score

class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels, init_weight=True, init_lrp=False):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        if init_weight:
            print("init_weight = True")
            def init_weights(module):
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                elif isinstance(module, BERTLayerNorm):
                    module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                    module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, nn.Linear):
                    module.bias.data.zero_()
            self.apply(init_weights)

        if init_lrp:
            print("init_lrp = True")
            init_hooks_lrp(self)

    def forward(self, input_ids, token_type_ids, attention_mask, seq_lens,
                device=None, labels=None):
        _, pooled_output, all_encoder_attention_scores, embedding_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, all_encoder_attention_scores, embedding_output
        else:
            return logits

    def backward_gradient(self, sensitivity_grads):
        classifier_out = func_activations['model.classifier']
        embedding_output = func_activations['model.bert.embeddings']
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output, 
                                                grad_outputs=sensitivity_grads, 
                                                retain_graph=True)[0]
        return sensitivity_grads
    
    def backward_gradient_input(self, sensitivity_grads):
        classifier_out = func_activations['model.classifier']
        embedding_output = func_activations['model.bert.embeddings']
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output, 
                                                grad_outputs=sensitivity_grads, 
                                                retain_graph=True)[0]
        return sensitivity_grads * embedding_output

    def backward_lrp(self, relevance_score):
        classifier_in = func_inputs['model.classifier'][0]
        classifier_out = func_activations['model.classifier']
        relevance_score = backprop_lrp_fc(self.classifier.weight,
                                          self.classifier.bias,
                                          classifier_in,
                                          relevance_score)
        relevance_score = self.bert.backward_lrp(relevance_score)
        return relevance_score
    
    def backward_lat(self, input_ids, attention_probs):
        
        # backing out using the quasi-attention
        attention_scores = torch.zeros_like(input_ids, dtype=torch.float)
        # we need to distribution the attention on CLS to each head
        # here, we use grad to do this
        attention_scores[:,0] = 1.0
        attention_scores = torch.stack(12 * [attention_scores], dim=1).unsqueeze(dim=2)

        for i in reversed(range(12)):
            attention_scores = torch.matmul(attention_scores, attention_probs[i])
        
        attention_scores = attention_scores.sum(dim=1).squeeze(dim=1).unsqueeze(dim=-1).data
        return attention_scores