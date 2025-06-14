import torch
import torch.nn as nn
import torch.utils.checkpoint

import math

from mlp import MultilayerPerceptron
from deeponets import DeepONetBase 


class VidonBranchNetSingleHead(nn.Module):
    """
    One transformer head, simplified structure compared to transformer paper.

    The last component of the last dimension of the input must be 0 for
    zero-padded points. This ensures that these points will have zero-weight.
    All other points can have value 1 (or any other constant).

    """
    def __init__(self, enc_dim, p, neurons_weights, neurons_values, act_fn, device=None):
        super().__init__()

        self.get_weights = MultilayerPerceptron(
            enc_dim, 1, neurons_weights, act_fn, device=device)

        self.get_values = MultilayerPerceptron(
            enc_dim, p, neurons_values, act_fn, device=device)

    def forward(self, x):
        inputs, mask = x

        weights = self.get_weights(inputs)
        weights /= math.sqrt(inputs.shape[-1])
        weights[mask] = -float('inf')  # zero weight for padded-points
        weights = torch.nn.Softmax(dim=1)(weights)

        values = self.get_values(inputs)

        prediction = torch.einsum('sf,sfp->sp', weights[..., 0], values)
        return prediction


class VidonBranchNet(nn.Module):
    """
    Manage all heads.

    """
    def __init__(self, coord_dim, sensor_dim, enc_dim, inner_dim, p, n_heads,
                 neurons_coord_enc, neurons_sensor_enc, neurons_weights,
                 neurons_values, neurons_combine, act_fn, device=None):

        super().__init__()

        self.enc_dim = enc_dim
        self.coord_dim = coord_dim
        self.inner_dim = inner_dim
        self.device = device

        self.lift_coords = MultilayerPerceptron(
            coord_dim, enc_dim, neurons_coord_enc, act_fn, device=device)

        self.lift_values = MultilayerPerceptron(
            sensor_dim, enc_dim, neurons_sensor_enc, act_fn, device=device)

        assert inner_dim % n_heads == 0, "'inner_dim' must be evenly divisible by 'n_heads'."

        self.heads = nn.ModuleList()

        for head in range(n_heads):
            self.heads.append(VidonBranchNetSingleHead(
                enc_dim, inner_dim // n_heads, neurons_weights, neurons_values, act_fn, device=device))

        self.combine = MultilayerPerceptron(
            inner_dim, p, neurons_combine, act_fn, device=device)

    def forward(self, x):
        coords, coord_inds, values = x
        n_samples, n_coords = values.shape[0], values.shape[1]

        # find padded points
        mask = (coord_inds == -1)

        # lift inputs
        lifted_coords = self.lift_coords(coords)
        lifted_values = self.lift_values(values)

        # for next step indices must not be negative
        coord_inds[mask] = 0

        # pick lifted coordinate for each sample
        coord_inds = coord_inds.view(n_samples * n_coords, 1)
        lifted_coords = torch.take_along_dim(lifted_coords, coord_inds, 0)
        lifted_coords = lifted_coords.view(n_samples, n_coords, self.enc_dim)

        inputs = lifted_coords + lifted_values

        res = torch.zeros(n_samples, self.inner_dim, device=self.device)
        h_size = self.inner_dim // len(self.heads)

        for hid, head in enumerate(self.heads):
            res[:, hid * h_size:(hid + 1) * h_size] = head((inputs, mask))

        return self.combine(res)

    
class Vidon(DeepONetBase):
    """
    Create a Variable-Input DeepONet.

    """
    def __init__(self, coord_in_dim, sensor_dim, encoded_branch_dim, inner_branch_dim, coord_out_dim, p, n_heads,
                 branch_neurons_coord_enc, branch_neurons_sensor_enc, branch_neurons_weights, branch_neurons_values,
                 branch_neurons_combine, trunk_neurons,
                 branch_act_fn=nn.Tanh(), trunk_act_fn=nn.Tanh(),
                 device=None):

        branch_net = VidonBranchNet(
            coord_in_dim, sensor_dim, encoded_branch_dim, inner_branch_dim, p, n_heads,
            branch_neurons_coord_enc, branch_neurons_sensor_enc,
            branch_neurons_weights, branch_neurons_values, branch_neurons_combine,
            branch_act_fn, device=device)
        
        trunk_net = MultilayerPerceptron(
            coord_out_dim, p, trunk_neurons, trunk_act_fn, device=device)

        super().__init__(p, branch_net, trunk_net)

    def forward_branch(self, x):
        """
        Overwritten from Base.
        
        """
        return self.branch_net(x)
    
    def forward(self, inputs):
        in_coords, in_coord_inds, sensor_values, out_coords, out_coord_inds = inputs
        branch_in = (in_coords, in_coord_inds, sensor_values)
        
        weights = self.forward_branch(branch_in)
        basis = self.forward_trunk(out_coords)

        # perform inner-product
        pred = torch.einsum('sp,xp->sx', weights, basis)[..., None]

        # pick coordinates belonging to each sample
        n_samples = len(out_coord_inds)
        n_coords = out_coord_inds.shape[1]

        mask = (out_coord_inds == -1)

        oci = out_coord_inds.detach().clone()
        oci[mask] = 0
        pred = torch.gather(pred, 1, oci)
        pred[mask] = 0

        del weights
        return pred
