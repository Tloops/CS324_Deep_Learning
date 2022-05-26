from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import torch
import torch.nn as nn


################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gx = nn.Linear(in_features=input_dim, out_features=hidden_dim).to(device)
        self.gh = nn.Linear(in_features=hidden_dim, out_features=hidden_dim).to(device)
        self.ix = nn.Linear(in_features=input_dim, out_features=hidden_dim).to(device)
        self.ih = nn.Linear(in_features=hidden_dim, out_features=hidden_dim).to(device)
        self.fx = nn.Linear(in_features=input_dim, out_features=hidden_dim).to(device)
        self.fh = nn.Linear(in_features=hidden_dim, out_features=hidden_dim).to(device)
        self.ox = nn.Linear(in_features=input_dim, out_features=hidden_dim).to(device)
        self.oh = nn.Linear(in_features=hidden_dim, out_features=hidden_dim).to(device)

        self.bg = torch.zeros(batch_size, hidden_dim).to(device)
        self.bi = torch.zeros(batch_size, hidden_dim).to(device)
        self.bf = torch.zeros(batch_size, hidden_dim).to(device)
        self.bo = torch.zeros(batch_size, hidden_dim).to(device)
        self.bp = torch.zeros(batch_size, output_dim).to(device)

        self.ph = nn.Linear(in_features=hidden_dim, out_features=output_dim).to(device)

    def forward(self, x):
        x = x.view(self.batch_size, -1, 1)

        self.h = torch.zeros(self.hidden_dim, self.hidden_dim)
        self.c = torch.zeros(self.hidden_dim, self.hidden_dim)
        if torch.cuda.is_available():
            self.h = self.h.cuda()
            self.c = self.c.cuda()
        for t in range(self.seq_length):
            g = torch.tanh(self.gx(x[:, t]) + self.gh(self.h) + self.bg)
            i = torch.sigmoid(self.ix(x[:, t]) + self.ih(self.h) + self.bi)
            f = torch.sigmoid(self.fx(x[:, t]) + self.fh(self.h) + self.bf)
            o = torch.sigmoid(self.ox(x[:, t]) + self.oh(self.h) + self.bo)
            self.c = g * i + self.c * f
            self.h = torch.tanh(self.c) * o
        return self.ph(self.h) + self.bp
