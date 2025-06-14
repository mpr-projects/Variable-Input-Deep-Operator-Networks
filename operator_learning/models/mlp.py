import torch
import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    r"""
    Create a multilayer perceptron. 

    """
    def __init__(self, n_in, n_out, neurons, act_fn, device=None, bias=True):
        super().__init__()

        self.ops = nn.ModuleList()
        self.act_fn = act_fn

        if isinstance(neurons, int):
            neurons = [neurons]

        # set up layers
        self.ops.append(nn.Linear(n_in, neurons[0], device=device, bias=bias))
        self.ops.append(self.act_fn)

        for nid, n in enumerate(neurons[1:]):
            self.ops.append(nn.Linear(neurons[nid], n, device=device, bias=bias))
            nn.init.xavier_normal_(self.ops[-1].weight)  # seems to work best for these models
            self.ops.append(self.act_fn)

        self.ops.append(nn.Linear(neurons[-1], n_out, device=device, bias=bias))


    def forward(self, x):
        for op in self.ops:
            x = op(x)
        return x
