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

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.inWeights = nn.Parameter(torch.nn.init.normal_(torch.empty( input_dim, num_hidden), 0, 0.5).to(device))
        self.hiddenWeights =  nn.Parameter(torch.nn.init.normal_(torch.empty(num_hidden, num_hidden), 0, 0.5).to(device))
        self.hiddenBias = nn.Parameter(torch.zeros(num_hidden).to(device))
        self.outWeights = nn.Parameter(torch.nn.init.normal_(torch.empty( num_hidden, num_classes), 0, 0.5).to(device))
        self.outBias = nn.Parameter(torch.zeros(num_classes).to(device))
        self.seq_length = seq_length
        self.hidden_size = num_hidden
        self.device = device
        self.store_hidden = False
        self.hiddenActivity = [None] * (seq_length-1)

    def forward(self, x):
        batchSize = x.size()[0]
        hiddenActivity = torch.zeros(batchSize, self.hidden_size).to(self.device)
        x = x.T
        for timeStep in range(self.seq_length-1):
            newIn = torch.matmul(x[timeStep].reshape(-1,1), self.inWeights)
            recurrent = torch.mm(hiddenActivity, self.hiddenWeights)
            hiddenActivity = torch.tanh(newIn + recurrent + self.hiddenBias) 
            if self.store_hidden:
                self.hiddenActivity[timeStep]=hiddenActivity
                self.hiddenActivity[timeStep].retain_grad()
        ## we don't need in this case to calc out at every step, but in general it could be useful
        out = torch.mm(hiddenActivity , self.outWeights) + self.outBias
        return out