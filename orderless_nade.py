# See: https://github.com/MarcCote/NADE/blob/a00c11bd23343012453711eb9d4b53614124122f/deepnade/buml/NADE/OrderlessBernoulliNADE.py#L154.

import torch

from torch import nn


class OrderlessNADE(nn.Module):
    def __init__(self, in_feats, mlp_layers):
        super().__init__()

        nade_mlp = nn.Sequential()
        in_feats = 2 * in_feats
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            nade_mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                nade_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            in_feats = out_feats

        self.nade_mlp = nade_mlp

    def forward(self, tensors):
        device = list(self.nade_mlp.parameters())[0].device

        (pixels, masks) = (tensors["pixels"], tensors["mask"])
        pixels = masks * pixels
        X = torch.cat([pixels, masks], dim=1)
        preds = self.nade_mlp(X.to(device))

        return preds
