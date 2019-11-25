################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        self.W_gx = nn.Parameter(torch.nn.init.normal_(torch.empty( input_dim, num_hidden), 0, 0.5).to(device))
        self.W_ix = nn.Parameter(torch.nn.init.normal_(torch.empty( input_dim, num_hidden), 0, 0.5).to(device))
        self.W_fx = nn.Parameter(torch.nn.init.normal_(torch.empty( input_dim, num_hidden), 0, 0.5).to(device))
        self.W_ox = nn.Parameter(torch.nn.init.normal_(torch.empty( input_dim, num_hidden), 0, 0.5).to(device))

        self.W_gh =  nn.Parameter(torch.nn.init.normal_(torch.empty(num_hidden, num_hidden), 0, 0.5).to(device))
        self.W_ih =  nn.Parameter(torch.nn.init.normal_(torch.empty(num_hidden, num_hidden), 0, 0.5).to(device))
        self.W_fh =  nn.Parameter(torch.nn.init.normal_(torch.empty(num_hidden, num_hidden), 0, 0.5).to(device))
        self.W_oh =  nn.Parameter(torch.nn.init.normal_(torch.empty(num_hidden, num_hidden), 0, 0.5).to(device))

        self.b_g = nn.Parameter(torch.zeros(num_hidden).to(device))
        self.b_i = nn.Parameter(torch.zeros(num_hidden).to(device))
        self.b_f = nn.Parameter(torch.zeros(num_hidden).to(device))
        self.b_o = nn.Parameter(torch.zeros(num_hidden).to(device))

        self.outWeights = nn.Parameter(torch.nn.init.normal_(torch.empty( num_hidden, num_classes), 0, 0.5).to(device))
        self.outBias = nn.Parameter(torch.zeros(num_classes).to(device))
        
        self.seq_length = seq_length
        self.hidden_size = num_hidden
        self.device = device

    def forward(self, x):
        batchSize = x.size()[0]
        hiddenActivity = torch.zeros(batchSize, self.hidden_size).to(self.device)
        state = torch.zeros(batchSize, self.hidden_size).to(self.device)

        x = x.T
        for timeStep in range(self.seq_length-1):
            inp = x[timeStep].reshape(-1,1)
            g = torch.tanh(torch.matmul(inp, self.W_gx) + torch.mm(hiddenActivity, self.W_gh) + self.b_g) 
            i = torch.sigmoid(torch.matmul(inp, self.W_ix) + torch.mm(hiddenActivity, self.W_ih) + self.b_i) 
            f = torch.sigmoid(torch.matmul(inp, self.W_fx) + torch.mm(hiddenActivity, self.W_fh) + self.b_f) 
            o = torch.sigmoid(torch.matmul(inp, self.W_ox) + torch.mm(hiddenActivity, self.W_oh) + self.b_o) 

            state = g*i + state*f
            hiddenActivity = torch.tanh(state) * o
        ## we don't need in this case to calc out at every step, but in general it could be useful
        out = torch.mm(hiddenActivity , self.outWeights) + self.outBias
        return out

