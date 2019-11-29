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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
from accuracies import accuracy, drawPlot
from genText import generate_text

################################################################################

def train(config):

    # Initialize the device which to run the model on
    device = torch.device('cuda:0')



    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset( config.txt_file, config.seq_length ) 
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, lstm_num_layers=2, device=device, temperature=config.temperature) 

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    accuracies = []

    batch_size = config.batch_size
    seq_length = config.seq_length
    vocab_size = dataset.vocab_size

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        optimizer.zero_grad()
        model.zero_grad()

    

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        ## input to one hot
     
        one_hot = torch.nn.functional.one_hot(batch_inputs, vocab_size).float()
        outputs = model(one_hot)
        outputs = outputs.flatten(0,1)
        batch_targets = batch_targets.flatten(0,1)
      
        loss = criterion(outputs, batch_targets)
        loss.backward()
        loss = loss.data.item()
        optimizer.step()
        outputs = outputs.cpu().detach().numpy()


        acc = accuracy(outputs, batch_targets.cpu().detach().numpy())
        accuracies.append(acc)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    int(step),
                    int(config.train_steps), 
                    config.batch_size, 
                    examples_per_second,
                    acc,
                    loss
            ))


        if step % config.sample_every == 0:
            max_indices = np.argmax(outputs, axis= 1)
            print(dataset.convert_to_string(max_indices))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    if config.generate_text:
        generate_text(model, dataset,  config.len_generation, device, stochastic=config.stochastic)           

    print('Done training.')
    drawPlot(accuracies, './LSTM_len:' +  '_lr:'+str(config.learning_rate)  + '_acc_text.jpg', "Accuracies on the next character with LSTM" , 1)


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='./part2/assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')


    # added params for text generation
    parser.add_argument('--generate_text', type=bool, default=True, help='Generate text after training using one letter')
    parser.add_argument('--len_generation', type=int, default=300, help='Length of generated text in characters')
    parser.add_argument('--temperature', type=float, default=1, help='Temperature for logit distribution')
    parser.add_argument('--stochastic', type=bool, default=True, help='Select the max probability, or sample stochastically from output')

    config = parser.parse_args()

    # Train the model
    train(config)
