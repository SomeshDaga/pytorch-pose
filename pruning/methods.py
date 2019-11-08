import numpy as np
import torch

from pruning.layers import MaskedConv2D
from pruning.utils import get_prune_stats

# def prune_one_kernel(model):
#     # Obtain references to all MaskedConv2D layers


#     # Determine the kernels with the smallest l2 norms
#     # and set their respective masks to 0
#     # NOTE: Ignore kernels that have already been eliminated
#     pass

def get_layer(model, name):
    path_list = name.split('.')
    item = model
    for path in path_list:
        if path.isdigit():
            item = item[int(path)]
        else:
            temp = getattr(item, path)
            if isinstance(temp, torch.nn.Module):
                item = temp
    return item

def prune_conv2d(model, pruning_percentage, prune_overall=True):
    '''
    Prune conv2d layers one by one until the pruning_percentage is achieved
    (NOTE: We are not performing iterative pruning)
    '''
    layer_list = []
    for name, param in model.named_parameters():
        # This is hacky, depends on naming conventions we have used
        if 'conv' in name and 'weight' in name:
            layer = get_layer(model, name)
            if isinstance(layer, MaskedConv2D):
                layer_list.append(layer)
                # Before attempting to prune weights, make sure that any existing mask values
                # have already been applied to the weights
                layer.apply_mask(None, None, None)

    num_removed = 0
    if prune_overall:
        if torch.cuda.is_available():
            all_norms = torch.Tensor().cuda()
        else:
            all_norms = torch.Tensor()
        for layer in layer_list:
            # Calculate l2 norms of each kernel for each layer
            norms = (layer.weight ** 2).sum(axis=1).sum(axis=1).sum(axis=1) / \
                (layer.weight.shape[1]*layer.weight.shape[2]*layer.weight.shape[3])
            # Normalize the norms
            norms = norms / torch.sqrt((norms ** 2).sum())

            # Concatenate the norms to the list of all norms
            all_norms = torch.cat((all_norms, norms.repeat(layer.weight.shape[1]*layer.weight.shape[2]*layer.weight.shape[3])))
        
        # Find the number of un-pruned filters/kernels
        total_filters = len(all_norms)
        non_zero_filters = torch.nonzero(all_norms).size(0)
        pruned_filters = total_filters - non_zero_filters

        # The percentage we want to prune by is w.r.t the un-pruned filters only
        prune_percentile = pruning_percentage + (pruned_filters * 100.0 / total_filters)
        
        # Find the threshold to remove the required percentage of weights
        threshold = np.percentile(all_norms.detach().cpu().numpy(), prune_percentile)

        for layer in layer_list:
            # Calculate l2 norms for all kernels in this layer
            norms = (layer.weight ** 2).sum(axis=1).sum(axis=1).sum(axis=1) / \
                (layer.weight.shape[1]*layer.weight.shape[2]*layer.weight.shape[3])

            # Normalize the norms
            norms = norms / torch.sqrt((norms ** 2).sum())

            # If a kernel norm is less than the threshold, set its corresponding mask value to False
            for i in range(0, len(layer.mask)):
                if norms[i] < threshold:
                    num_removed = num_removed + 1
                    layer.mask[i] = False

    else:
        for layer in layer_list:
            # Calculate l2 norms of each kernel for each layer
            norms = (layer.weight ** 2).sum(axis=1).sum(axis=1).sum(axis=1) / \
                (layer.weight.shape[1]*layer.weight.shape[2]*layer.weight.shape[3])

            # Normalize the norms
            norms = norms / torch.sqrt((norms ** 2).sum())

            # Find number of trainable kernels remaining in the layer
            # that we want to eliminate
            non_zero_filters = torch.nonzero(norms).size(0)
            total_filters = len(norms)
            pruned_filters = total_filters - non_zero_filters
            
            prune_percentile = pruning_percentage + (pruned_filters * 100.0/ total_filters)
            threshold = np.percentile(norms.detach().cpu().numpy(), prune_percentile)

            # If a kernel norm is less than the threshold, set its corresponding mask value to False
            for i in range(0, len(layer.mask)):
                if norms[i] < threshold:
                    num_removed = num_removed + 1
                    layer.mask[i] = False

    # Ensure weights in all layers are updated with changes to masks
    total_conv_weights = 0
    for layer in layer_list:
        total_conv_weights += layer.weight.numel()     
        layer.apply_mask(None, None, None)

    print("Num removed: {0}, Total initial weights: {1}".format(num_removed, total_conv_weights))

    # Get the stats on the pruned model
    get_prune_stats(model)

    return
