"""
  FileName     [ classifier.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Fully connected models for predict labels ]
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class Classifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=[2048], num_class=11, norm_layer=nn.BatchNorm1d, activation=nn.ReLU(inplace=True), dropout=0):
        super(Classifier, self).__init__()

        self.norm_layer = norm_layer
        self.activation = activation
        self.dropout    = dropout

        self.fcs    = self.make_layer(feature_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim[-1], num_class)

    def make_layer(self, input_dim, dims):
        layers = []

        for dim in dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.norm_layer(dim))
            layers.append(self.activation)
            
            if self.dropout:
                layers.append(self.dropout)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fc_out(x)

        return x
