# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, target=None):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        if target is None or not self.training:
            # [batch_size, n_out, seq_len, seq_len]
            s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        else:
            # [batch_size, seq_len, seq_len, n_labels] (sparse)
            if not target.is_sparse:
                target = target.to_sparse(3)
            b_idx, x_idx, y_idx = target.indices()
            # [*, n_in]
            x = x[b_idx, x_idx]
            y = y[b_idx, y_idx]
            # [batch_size, n_out, seq_len, seq_len]
            s = torch.einsum('ni,oij,nj->no', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)
        return s
