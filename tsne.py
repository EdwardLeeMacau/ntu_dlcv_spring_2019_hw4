"""
  FileName     [ tsne.py ]
  PackageName  [ HW4 ]
  Synopsis     [ HW4-1 t-SNE video feature visualiztion ]
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader

import utils
from classifier import Classifier
from dataset import TrimmedVideos
from rnn import LSTM_Net

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default="./model/problem2.pth", type=str, help='The directory of model to load.')
parser.add_argument("--graph", default="./output", type=str, help='The folder to save the tsne figure')
parser.add_argument("--video", default="./hw4_data/TrimmedVideos/video/valid", type=str, help='The directory of the videos')
parser.add_argument("--label", default="./hw4_data/TrimmedVideos/label/gt_valid.csv", type=str, help='The directory of the ground truth label')
parser.add_argument("--feature", default="./hw4_data/TrimmedVideos/feature/valid" ,type=str, help='The path of the features')
parser.add_argument("--plot_num", default=512, type=int, help='The number of points in the graphs')
parser.add_argument("--plot_size", default=20, type=int, help="The size of points in the graphs")
opt = parser.parse_args()

DEVICE = utils.selectDevice()

def dimension_reduction_cnn(fname, loader, model=None):
    if model:
        model.eval()

    with torch.no_grad():
        dataiter = iter(loader)

        features, labels = dataiter.next()
        features = features.view(-1, 8192)
        labels   = labels.view(-1)
        print('labels.shape:   {}'.format(labels.shape))
        print('Features.shape: {}'.format(features.shape))
        
        tsne = TSNE(n_components=2)
        embedded = tsne.fit_transform(features)
        embedded = torch.from_numpy(embedded)

        print('embedded.shape: ', embedded.shape)
        print('plot_num: ', opt.plot_num)
        print('labels.shape :', labels.shape)

        plot_features(fname, embedded, labels, opt.plot_num)
    
    return

def dimension_reduction_rnn(fname, loader, model=None):
    if model:
        model.eval()

    with torch.no_grad():
        dataiter = iter(loader)

        features = torch.zeros((opt.plot_num, 128), dtype=torch.float32)
        labels   = torch.zeros(opt.plot_num, dtype=torch.float32)

        for index, (feature, label) in enumerate(loader, 0):
            if index == opt.plot_num: break

            feature = feature.permute(1, 0, 2)
            # print(feature.shape)
            
            feature   = pack_padded_sequence(feature, [len(feature)], batch_first=False).to(DEVICE)
            _, (h, c) = model(feature)
            h         = h[-1, 0].cpu()# .detach().data.numpy()

            features[index] = h
            labels[index]   = label
        
        tsne = TSNE(n_components=2)
        embedded = tsne.fit_transform(features)
        embedded = torch.from_numpy(embedded)

        print('features_embedded.shape: ', embedded.shape)
        print('plot_num: ', opt.plot_num)
        print('labels.shape :', labels.shape)

        plot_features(fname, embedded, labels, opt.plot_num)
    
    return

def plot_features(fname, features: np.ndarray, labels: np.ndarray, plot_num, title=""):
    """
    Parameters
    ----------
    fname : str
        saved filename

    features : 
        dim [n, 2]
    
    labels : 
        dim [n]
    
    plot_num : 
    """
    colors = plt.get_cmap('Set1')

    plt.figure(figsize=(12.8, 7.2))

    for num in range(11):
        mask_target = (labels == num)
        x, y = features[:, 0], features[:, 1]
        x = torch.masked_select( x, mask_target )
        y = torch.masked_select( y, mask_target )
        
        plt.scatter(x, y, s=opt.plot_size, c=colors(num), alpha=0.6, label=str(num))
        
    plt.title(title)
    plt.legend(loc=0)
    plt.savefig(fname)

    plt.close('all')

    return

def main():
    valids_p1 = TrimmedVideos(None, opt.label, opt.feature, sample=4, transform=transforms.ToTensor())
    loader_p1 = DataLoader(valids_p1, batch_size=opt.plot_num, shuffle=False)

    valids_p2 = TrimmedVideos(None, opt.label, opt.feature, downsample=12, transform=transforms.ToTensor())
    loader_p2 = DataLoader(valids_p2, batch_size=1, shuffle=False)

    recurrent = utils.loadModel(opt.resume, 
                    LSTM_Net(2048, 128, 11, num_layers=2, bias=True, 
                    dropout=0.2, bidirectional=False, seq_predict=False)
                ).to(DEVICE)

    graph_1 = os.path.join(opt.graph, 'p1_tsne.png')
    graph_2 = os.path.join(opt.graph, 'p2_tsne.png')

    dimension_reduction_cnn(graph_1, loader_p1)
    dimension_reduction_rnn(graph_2, loader_p2, recurrent)

if __name__ == '__main__':   
    main()
