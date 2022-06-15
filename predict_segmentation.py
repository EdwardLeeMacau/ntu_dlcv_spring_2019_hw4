"""
  FileName     [ predict_segmentation.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Generate the video action segmentation based on the RNN model. ]
"""

import argparse
import logging
import logging.config
import os
import random
import time

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import utils
from cnn import resnet50
from rnn import LSTM_Net

parser = argparse.ArgumentParser()

# Basic Training setting
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--downsample", default=1, type=int, help="the downsample ratio of the training data.")
parser.add_argument("--dropout", default=0.2, help="the dropout probability of the recurrent network")
# Model dimension setting
parser.add_argument("--layers", default=2, help="the number of the recurrent layers")
parser.add_argument("--bidirection", default=False, action="store_true", help="Use the bidirectional recurrent network")
parser.add_argument("--hidden_dim", default=512, help="the dimension of the RNN's hidden layer")
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Devices setting
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--max_length", type=int, default=500, help="number of image to read at a time")
# Load dataset, pretrain model setting
parser.add_argument("--resume", default="./model/problem3.pth", type=str, help="path to load the model")
parser.add_argument("--video", default="./hw4_data/FullLengthVideos/videos/valid", type=str, help="path to the videos directory")
parser.add_argument("--label", default="./hw4_data/FullLengthVideos/labels/valid", type=str, help="path of the label csv file")
parser.add_argument("--output", default="./output", help="The predict textfile path.")

opt = parser.parse_args()

# Set as true when the I/O shape of the model is fixed
DEVICE = utils.selectDevice()

def predict(extractor, model, loader):
    """ Predict the model's performance. """
    extractor.eval()
    model.eval()

    print("[{}]".format(time.asctime()))
    with torch.no_grad():
        for index, (video, _, video_name) in enumerate(loader, 1):
            # Extract the features
            features = torch.zeros((video.shape[0], 2048), dtype=torch.float32).to(DEVICE)
            remain, finish = video.shape[0], 0
            while remain:
                step = min(remain, opt.max_length)
                todo = video[finish : finish + step].to(DEVICE)
                features[finish : finish + step] = extractor(todo)
                remain -= step
                finish += step

            # Predict the sequences
            predict = model(features).argmax(dim=1).cpu().tolist()

            if index == 1:
                print("The process allocated GPU with {:.1f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))

            np.savetxt(os.path.join(opt.output, video_name + '.txt'), predict, fmt='%d')

            print("[{}][Predict][ {:4d}/{:4d} ][{:.2%}][{}]".format(
                time.asctime(), index, len(loader.dataset), index / len(loader.dataset), features.shape[0])
            )
    
            print("Output File have been written to {}".format(os.path.join(opt.output, video_name + '.txt')))

    return 

def main():
    extractor = resnet50(pretrained=True).to(DEVICE)
    recurrent = utils.loadModel(opt.model, 
                    LSTM_Net(2048, opt.hidden_dim, opt.output_dim, 
                    num_layers=opt.layers, bias=True, dropout=opt.dropout, 
                    bidirectional=opt.bidirectional, seq_predict=False)
                ).to(DEVICE)
    
    predict_set = dataset.TrimmedVideos(opt.video, opt.label, None, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    
    print("Dataset: {}".format(len(predict_set)))
    predict_loader = DataLoader(predict_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    # Predict
    predict(extractor, recurrent, predict_loader)

if __name__ == "__main__":
    os.system("clear")
    
    for key, value in vars(opt).items():
        print("{:15} {}".format(key, value))
    
    main()
