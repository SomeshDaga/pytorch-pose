import numpy as np
import torch
from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False):
    """
    Return a torch.autograd.Variable based on system requirements
    """
    if torch.cuda.is_available():
        x = x.cuda()
    else:
        x.to('cpu')
    
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def get_prune_stats(model):
    total_parameters_initial = sum(p.numel() for p in model.parameters())
    total_non_zero_parameters = sum(np.count_nonzero(p.data.cpu().numpy()) for p in model.parameters())
    print("Percentage Pruned: {0}%".format((total_parameters_initial - total_non_zero_parameters) * 100.0 / total_parameters_initial))
    print("Num initial parameters: {0}, Num reduced parameters: {1}".format(total_parameters_initial, total_non_zero_parameters))
