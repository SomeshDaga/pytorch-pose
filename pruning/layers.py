import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from pruning.utils import to_var

class MaskedConv2D(nn.Conv2d):
    """
    A "Masked" version of the PyTorch 2D Convolutional Layer where weights can be dynamically removed
    from the network by making them untrainable using a mask
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2D, self).__init__(in_channels, out_channels, 
                                          kernel_size, stride, padding, dilation, groups, bias)
        # Register a mask buffer to ensure it gets stored in the state_dict at the checkpoints
        # We register it as a buffer instead of a parameter because the mask does not get trained
        # Note: The size of the mask is equal to the number of distinct kernels, since
        #       we will either completely use a kernel or completely discard it
        self.register_buffer('mask',
            torch.ones(self.weight.data.shape[0]).type(torch.BoolTensor))
        
        self.register_backward_hook(self.apply_mask)
    
    def set_mask(self, mask):
        # Update the buffer value for the mask
        self.mask = mask
    
    def get_mask(self):
        return self.mask

    def apply_mask(self, input_grad, output_grad, extra):
        '''
        Resets weights to 0 for those kernels with a corresponding null mask
        '''
        # print "Num non-zero mask values: {0}".format(torch.nonzero(self.weight.data).size(0))
        # print "Any false: {0}".format("yes" if any(self.mask) is False else "no")
        # print self.mask.size()
        if not all(self.mask):
            # If a mask exists, perform an element-wise multiplication of the weights with the mask
            # such that mask elements of 0 render the corresponding weight as untrainable and hence
            # reduce the number of parameters in the convolutional layer
            self.weight.data = self.weight.data * \
                self.mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # print "Num non-zero mask values: {0}".format(torch.nonzero(self.weight.data).size(0))

    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Key values in the state dictionary for parameters/buffers are prefixed
        # with the layer path that are unique for each layer in the network
        mask_key = prefix + "mask"
        if mask_key not in state_dict:
            state_dict[mask_key] = torch.ones(self.weight.data.shape[0]).type(torch.BoolTensor)

        super(MaskedConv2D, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
