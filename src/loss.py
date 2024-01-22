import torch
import torch.nn.functional as F

def balanced_cross_entropy(input, target):
    batch, _, width, height = target.size()
    pos_index = (target >=0.5)
    neg_index = (target < 0.5)    
    weight = torch.zeros_like(target)
    sum_num = width*height
    pos_num = pos_index.sum().item()
    neg_num = sum_num - pos_num
    weight[pos_index] = neg_num / sum_num
    weight[neg_index] = pos_num / sum_num
    loss = F.binary_cross_entropy(input, target, weight, reduction='none')

    return torch.sum(loss)/batch