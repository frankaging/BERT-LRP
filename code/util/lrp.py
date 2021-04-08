import torch
import torch.nn as nn
import numpy as np
from numpy import newaxis as na


##############################################################################
#
# The function to back-out layerwise attended relevance scores.
#
##############################################################################
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

def backprop_lrp_nl(weight, activations, R, 
                    eps=1e-7, alpha=0.5, debug=False):
    """
    This is for non-linear linear lrp.
    We use jacobian and first term of Taylor expansions.
    weight: [b, l, h_out, h_in]
    activations: [b, l, h_in]
    R: [b, l, h_out]
    """
    beta = 1.0 - alpha
    R = R.unsqueeze(dim=2) # [b, l, 1, h_out]
    activations = activations.unsqueeze(dim=2) # [b, l, 1, h_in]

    weight_p = torch.clamp(weight, min=0.0) 
    z_p = torch.matmul(activations, weight_p.transpose(2,3)) + eps
    s_p = R / z_p # [b, l, 1, h_out]
    c_p = torch.matmul(s_p, weight_p) # [b, l, 1, h_in]
    
    weight_n = torch.clamp(weight, max=0.0)
    z_n = torch.matmul(activations, weight_n.transpose(2,3)) + eps 
    s_n = R / z_n
    c_n = torch.matmul(s_n, weight_n)

    R_c = activations * (alpha * c_p + beta * c_n)
    
    R_c = R_c.squeeze(dim=2)
    R = R.squeeze(dim=2)
    R_c = rescale_lrp(R, R_c)

    return R_c

def rescale_jacobian(output_relevance, *input_relevances, batch_axes=(0,)):
    assert isinstance(batch_axes, (tuple, list))
    get_summation_axes = lambda tensor: tuple(i for i in range(len(tensor.shape)) if i not in batch_axes)
    ref_scale = abs(output_relevance).sum(dim=get_summation_axes(output_relevance), keepdim=True)
    inp_scales = [abs(inp).sum(dim=get_summation_axes(inp), keepdim=True) for inp in input_relevances]
    total_inp_scale = sum(inp_scales) + 1e-7
    input_relevances = [inp * (ref_scale / total_inp_scale) for inp in input_relevances]
    return input_relevances[0] if len(input_relevances) == 1 else input_relevances

def backprop_lrp_jacobian(jacobians, output, R, inps, eps=1e-7, alpha=0.5, batch_axes=(0,)):
    """
    computes input relevance given output_relevance using z+ rule
    works for linear layers, convolutions, poolings, etc.
    notation from DOI:10.1371/journal.pone.0130140, Eq 60
    """
    
    beta = 1.0 - alpha
    inps = [inp for inp in inps]

    reference_inputs = tuple(map(torch.zeros_like, inps))
    assert len(reference_inputs) == len(inps)

    flat_output_relevance = R.reshape([-1])
    output_size = flat_output_relevance.shape[0]

    assert len(jacobians) == len(inps)

    jac_flat_components = [jac.reshape([output_size, -1]) for jac in jacobians]
    # ^-- list of [output_size, input_size] for each input
    flat_jacobian = torch.cat(jac_flat_components, dim=-1)  # [output_size, combined_input_size]

    # 2. multiply jacobian by input to get unnormalized relevances, add bias
    flat_input = torch.cat([inp.reshape([-1]) for inp in inps], dim=-1)  # [combined_input_size]
    flat_reference_input = torch.cat([ref.reshape([-1]) for ref in reference_inputs], dim=-1)
    import operator
    from functools import reduce 
    num_samples = reduce(operator.mul, [output.shape[batch_axis] for batch_axis in batch_axes], 1)
    input_size_per_sample = flat_reference_input.shape[0] // num_samples
    flat_impact = (flat_jacobian * flat_input[None, :])
    # ^-- [output_size, combined_input_size], aka z_{j<-i}

    # 3. normalize positive and negative relevance separately and add them with coefficients
    flat_positive_impact = torch.clamp(flat_impact, min=0.0)
    flat_positive_normalizer = flat_positive_impact.sum(dim=0, keepdim=True) + eps
    flat_positive_relevance = flat_positive_impact / flat_positive_normalizer

    flat_negative_impact = torch.clamp(flat_impact, max=0.0)
    flat_negative_normalizer = flat_negative_impact.sum(dim=0, keepdim=True) - eps
    flat_negative_relevance = flat_negative_impact / flat_negative_normalizer
    flat_total_relevance_transition = alpha * flat_positive_relevance + beta * flat_negative_relevance

    flat_input_relevance = torch.einsum('o,oi->i', flat_output_relevance, flat_total_relevance_transition)
    # ^-- [combined_input_size]

    # 5. unpack flat_inp_relevance back into individual tensors
    input_relevances = []
    offset = 0
    for inp in inps:
        inp_size = inp.reshape([-1]).shape[0]
        inp_relevance = flat_input_relevance[offset: offset + inp_size].reshape(inp.shape)
        inp_relevance = inp_relevance.contiguous()
        input_relevances.append(inp_relevance)
        offset = offset + inp_size
    
    return rescale_jacobian(R, *input_relevances, batch_axes=batch_axes)