from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.hx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        # Implementation here ...
        hs = []
        for i in range(self.seq_length - 1):
            h = torch.zeros(self.batch_size, self.hidden_dim, device=x.device)
            for j in range(self.batch_size):
                if i == 0:
                    h[j] = torch.tanh(self.hx(x[j][i].view(-1)))
                else:
                    h[j] = torch.tanh(self.hx(x[j][i].view(-1)) + self.hh(hs[i - 1][j]))
            hs.append(h)
        return self.out(hs[-1])
    # add more methods here if needed
