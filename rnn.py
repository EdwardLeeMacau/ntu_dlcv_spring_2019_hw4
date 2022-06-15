"""
  FileName     [ rnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Test RNN, LSTM, bidirectional LSTM and GRU model ]
"""

import argparse
import datetime
import os
import random
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

import dataset
import utils
from cnn import resnet50

DEVICE = utils.selectDevice()

class LSTM_Net(nn.Module):
    """ The model to learn the video information by LSTM kernel """
    def __init__(self, feature_dim, hidden_dim, output_dim, num_layers=1, bias=True, batch_first=False, 
                 dropout=0, bidirectional=False, seq_predict=False):
        """
          Params:
          - feature_dim: dimension of the feature vector extracted from frames
          - hidden_dim:  dimension of the hidden layer
          - output_dim:  dimension of the classifier output
        """
        super(LSTM_Net, self).__init__()

        self.feature_dim   = feature_dim
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.num_layer     = num_layers
        self.bias          = bias
        self.batch_first   = batch_first
        self.dropout       = dropout
        self.bidirectional = bidirectional
        self.seq_predict   = seq_predict

        self.recurrent  = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers, bias=bias, batch_first=batch_first, 
                                  dropout=dropout, bidirectional=bidirectional)
        
        if self.bidirectional:
            hidden_dim *= 2

        self.fc_out     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, h=None, c=None):
        """
          Params:
          - x: the tensor of frames
               if batch_first: (batchsize, length, feature_dim)
               if not:         (length, batchsize, feature_dim)
          
          Return:
          - x: the class prediction tensor: 
               if seq_predict: (length, batch, output_dim)
               if not:         (batch, output_dim)
        """
        # -------------------------------------------------------------------
        # lstm_out, hidden_state, cell_state = LSTM(x, (hidden_state, cell_state))
        # -> lstm_out is the hidden_state tensor of the highest lstm cell.
        # -------------------------------------------------------------------
        if (h is not None) and (c is not None):
            x, (h, c) = self.recurrent(x, (h, c))
        else:
            x, (h, c) = self.recurrent(x)
        x, seq_len = pad_packed_sequence(x, batch_first=self.batch_first)
        
        # --------------------------------------------
        # Sequence-to-1 prediction:
        #   get the last 1 output if only it is needed.
        # --------------------------------------------
        if not self.seq_predict:
            if self.batch_first:
                x = x[:,-1]
            else:
                x = x[-1]
        
        # -------------------------------------------
        # Sequence-to-Sequence prediction:
        #   get the output with the whole series
        # -------------------------------------------
        if self.seq_predict:
            length = x.shape[0]
            
            if self.bidirectional:
                x = x.view(-1, 2 * self.hidden_dim)
            else:
                x = x.view(-1, self.hidden_dim)

        x = self.fc_out(x)

        if self.seq_predict:
            x = x.view(length, -1, self.output_dim)

        return x, (h, c)

def main():
    torch.manual_seed(1)

    model = LSTM_Net(2048, 128, 11, num_layers=1, batch_first=False)

    # Way to initial the model parameters
    for param in model.recurrent.parameters():
        if len(param.shape) >= 2:
            nn.init.orthogonal_(param)
        

if __name__ == "__main__":
    main()
