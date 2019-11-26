# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0', temperature = 1):

        super(TextGenerationModel, self).__init__()
        self.lstm_layers = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers, batch_first=True, dropout=0.2).to(device)
        #self.classifier = nn.Sequential(nn.Linear(lstm_num_hidden, 128),nn.ReLU(),nn.Linear(128, 64),nn.ReLU(),nn.Linear(64, vocabulary_size)).to(device)
        #self.classifier = nn.Linear(lstm_num_hidden, vocabulary_size).to(device)
        self.classifier = nn.Sequential(nn.Linear(lstm_num_hidden, 128),nn.ReLU(),nn.Linear(128, vocabulary_size)).to(device)
        self.temperature = temperature


    def forward(self, x):
        out, (hidden, state) = self.lstm_layers(x)
        out = self.classifier(out)
        out = out*self.temperature
        return out
