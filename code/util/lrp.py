import torch
import torch.nn as nn
import numpy as np
from numpy import newaxis as na


##############################################################################
#
# The function to back-out layerwise attended relevance scores.
#
##############################################################################

def backprop_lrp_fc(weight, bias, activations, R, 
                    eps=1e-7, alpha=0.5, debug=False):
    beta = 1.0 - alpha
    
    weight_p = torch.clamp(weight, min=0.0)
    bias_p = torch.clamp(bias, min=0.0)    
    z_p = torch.matmul(activations, weight_p.T) + bias_p + eps
    s_p = R / z_p
    c_p = torch.matmul(s_p, weight_p)
    
    weight_n = torch.clamp(weight, max=0.0)
    bias_n = torch.clamp(bias, max=0.0)
    z_n = torch.matmul(activations, weight_n.T) + bias_n - eps 
    s_n = R / z_n
    c_n = torch.matmul(s_n, weight_n)

    R_c = activations * (alpha * c_p + beta * c_n)
    
    R_c = rescale_lrp(R, R_c)

    return R_c

def rescale_lrp(post_A, inp_relevances):
    inp_relevances = torch.abs(inp_relevances)
    if len(post_A.shape) == 2:
        ref_scale = torch.sum(post_A, dim=-1, keepdim=True) + 1e-7
        inp_scale = torch.sum(inp_relevances, dim=-1, keepdim=True) + 1e-7
    elif len(post_A.shape) == 3:
        ref_scale = post_A.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + 1e-7
        inp_scale = inp_relevances.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + 1e-7
    scaler = ref_scale / inp_scale
    inp_relevances = inp_relevances * scaler
    return inp_relevances