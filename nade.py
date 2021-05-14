# See: https://github.com/MarcCote/NADE/blob/a00c11bd23343012453711eb9d4b53614124122f/deepnade/buml/NADE/OrderlessBernoulliNADE.py#L154.

import torch

from torch import nn


class NADE(nn.Module):
    def __init__(self, in_feats, hidden_dim):
        super().__init__()
        self.register_parameter("W", nn.Parameter(torch.randn(hidden_dim, in_feats)))
        self.register_parameter("c", nn.Parameter(torch.randn(hidden_dim)))
        self.register_parameter("V", nn.Parameter(torch.randn(in_feats, hidden_dim)))
        self.register_parameter("b", nn.Parameter(torch.randn(in_feats)))

    def forward(self, tensors):
        device = self.c.device

        v = tensors["pixels"].to(device)
        a = self.c.unsqueeze(0).repeat(len(v), 1)
        preds = []
        for i in range(len(v[0])):
            h_i = torch.sigmoid(a)
            preds.append(self.b[i] + self.V[i].unsqueeze(0) @ h_i.T)
            a += self.W[:, i].unsqueeze(0) * v[:, i].unsqueeze(1)

        preds = torch.vstack(preds).T

        return preds
