import torch
import torch.nn as nn

from mlp import MultilayerPerceptron
from debug_tools import format_tensor_size


class DeepONetBase(nn.Module):
    """
    Not suitable for RNN version! Must not have additional dimensions in the trunk.
    Output is always scalar-valued.
    
    """
    def __init__(self, p, branch_net, trunk_net):
        super().__init__()
        self.p = p
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')
    
    def forward_trunk(self, x):
        """
        Evaluate trunk net. Input shape: (n_points, dim).
        
        """
        # ensure there are exactly two dimensions even if input
        # only has one, e.g. if input has format: (n_points)
        x = x.reshape(x.shape[0], -1)
        
        # basis.shape: (n_xpoints, p)
        return self.trunk_net(x)
    
    def forward_branch(self, x):
        """
        Evaluate branch net. Input shape: (n_samples, n_features).
        
        """
        n_samples = x.shape[0]
        
        # ensure there are at exactly two dimensions even if input
        # only has one or three, e.g. if input has format: (n_samples)
        # or (n_samples, n_xpoints, 1)
        x = x.contiguous()
        x = x.view(n_samples, -1)
        
        # weights.shape = (n_samples, p)
        return self.branch_net(x)

    def forward(self, inputs):
        # format of inputs:
        #   branch_in: (n_samples, n_features)
        #   trunk_in: (n_points, dim)
        branch_in, trunk_in = inputs
        
        weights = self.forward_branch(branch_in)
        basis = self.forward_trunk(trunk_in)

        # perform inner-product
        pred = torch.einsum('sp,xp->sx', weights, basis)[..., None]

        del weights
        return pred
    

class MLPDeepONet(DeepONetBase):
    
    def __init__(self, p,
                 branch_n_in, trunk_n_in,
                 branch_neurons, trunk_neurons,
                 branch_act_fn=nn.Tanh(), trunk_act_fn=nn.Tanh(),
                 device=None):
        
        branch_net = MultilayerPerceptron(
            branch_n_in, p, branch_neurons, branch_act_fn, device=device)
        
        trunk_net = MultilayerPerceptron(
            trunk_n_in, p, trunk_neurons, trunk_act_fn, device=device)
        
        super().__init__(p, branch_net, trunk_net)
