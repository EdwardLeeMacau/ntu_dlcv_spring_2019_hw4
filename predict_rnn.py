"""
  FileName     [ predict_rnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Generate the prediction action labels based on the RNN model. ]
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
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import utils
from cnn import resnet50
from rnn import LSTM_Net

parser = argparse.ArgumentParser()

# Basic Training setting
parser.add_argument("--batch_size", default=1, type=int, help="size of the batches")
parser.add_argument("--downsample", default=12, type=int, help="the downsample ratio of the training data.")
parser.add_argument("--dropout", default=0.2, type=float, help="the dropout probability of the recurrent network")
# Model dimension setting
parser.add_argument("--layers", default=2, type=int, help="the number of the recurrent layers")
parser.add_argument("--bidirectional", default=False, action="store_true", help="Use the bidirectional recurrent network")
parser.add_argument("--hidden_dim", default=128, type=int, help="the dimension of the RNN's hidden layer")
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Devices setting
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--max_length", type=int, default=500, help="number of image to read at a time")
# Load dataset, pretrain model setting
parser.add_argument("--resume", default="./model/problem2.pth", type=str, help="path to load the model")
parser.add_argument("--video", default="./hw4_data/TrimmedVideos/video/valid", type=str, help="path to the videos directory")
parser.add_argument("--label", default="./hw4_data/TrimmedVideos/label/gt_valid.csv", type=str, help="path of the label csv file")
parser.add_argument("--output", default="./output", help="The predict textfile path.")

opt = parser.parse_args()

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def predict(extractor, model, loader):
    """ Predict the model's performance. """
    extractor.eval()
    model.eval()

    results = []
    predict_done = 0
    with torch.no_grad():
        for index, (video, _, seq_len) in enumerate(loader, 1):
            batchsize = len(seq_len)
            
            predict_done += batchsize
            print("[{}][Predict][ {:4d}/{:4d} ][{:.2%}][{}]".format(
                time.asctime(), predict_done, len(loader.dataset), predict_done / len(loader.dataset), seq_len)
            )

            # Read the images with the limited size
            features = torch.zeros((video.shape[0], 2048), dtype=torch.float32).to(DEVICE)
            remain, finish = video.shape[0], 0
            while remain:
                step = min(remain, opt.max_length)
                todo = video[finish : finish + step].to(DEVICE)
                features[finish : finish + step] = extractor(todo)
                remain -= step
                finish += step

            features = pad_sequence(torch.split(features, seq_len, dim=0), batch_first=False)
            features = pack_padded_sequence(features, seq_len, batch_first=False)
            predict, _ = model(features)
            predict = predict.argmax(dim=1).cpu().tolist()

            if index == 1:
                print("The process allocated GPU with {:.1f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))

            results.extend(predict)
            
    return results

def main():
    opt.output = os.path.join(opt.output, 'p2_result.txt')

    extractor = resnet50(pretrained=True).to(DEVICE)
    recurrent = utils.loadModel(opt.resume, 
                    LSTM_Net(2048, opt.hidden_dim, opt.output_dim, 
                    num_layers=opt.layers, bias=True, dropout=opt.dropout, 
                    bidirectional=opt.bidirectional, seq_predict=False)
                ).to(DEVICE)
    
    predict_set = dataset.TrimmedVideos(opt.video, opt.label, None, downsample=opt.downsample, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    
    print("Dataset: {}".format(len(predict_set)))
    predict_loader = DataLoader(predict_set, batch_size=opt.batch_size, shuffle=False, collate_fn=utils.collate_fn_valid, num_workers=opt.threads)
    
    # Predict
    results = predict(extractor, recurrent, predict_loader)
    np.savetxt(opt.output, results, fmt='%d')
    print("Output File have been written to {}".format(opt.output))

if __name__ == "__main__":
    os.system("clear")
    
    for key, value in vars(opt).items():
        print("{:15} {}".format(key, value))
    
    main()
