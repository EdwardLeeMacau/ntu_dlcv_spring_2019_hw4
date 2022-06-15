"""
  FileName     [ predict_cnn.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Generate the prediction action labels based on the CNN model. ]
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
from classifier import Classifier

parser = argparse.ArgumentParser()

# Basic Training setting
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--downsample", default=12, type=int, help="the downsample ratio of the training data.")
# Model dimension setting
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Devices setting
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--resume", default="./model/problem1.pth", type=str, help="path to load the model")
parser.add_argument("--video", default="./hw4_data/TrimmedVideos/video/valid", type=str, help="path to the videos directory")
parser.add_argument("--label", default="./hw4_data/TrimmedVideos/label/gt_valid.csv", type=str, help="path of the label csv file")
parser.add_argument("--output", default="./output", help="The predict csvfile path.")

opt = parser.parse_args()

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def predict(extractor: nn.Module, model: nn.Module, loader: DataLoader) -> np.array:
    """ Predict the model's performance. """
    extractor.eval()
    model.eval()

    result = []

    print("[{}]".format(time.asctime()))

    predict_done = 0
    with torch.no_grad():
        for index, (video, _) in enumerate(loader, 1):
            batchsize = video.shape[0]

            video      = video.view(-1, 3, 240, 320).to(DEVICE)
            feature    = extractor(video).view(batchsize, -1)
            predict    = model(feature).argmax(dim=1)
            # video_name = [name.split("/")[-1] for name in video_name]

            if index == 1:
                print("The process allocated GPU with {:.1f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))

            result.extend(predict)
            predict_done += batchsize

            print("[{}][Predict][ {:4d}/{:4d} ][{:.2%}]".format(time.asctime(), predict_done, len(loader.dataset), predict_done / len(loader.dataset)))

    return np.array(result).astype(int).transpose()

def main():
    if not os.path.exists(opt.output):
        os.makedirs(opt.output, exist_ok=True)

    opt.output = os.path.join(opt.output, 'p1_valid.txt')

    extractor  = resnet50(pretrained=True)
    classifier = utils.loadModel(opt.resume, Classifier(8192, [2048], 11))
    extractor, classifier = extractor.to(DEVICE), classifier.to(DEVICE)
    
    predict_set = dataset.TrimmedVideos(opt.video, opt.label, None, sample=4, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print("Dataset: {}".format(len(predict_set)))
    predict_loader = DataLoader(predict_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    # Predict
    results = predict(extractor, classifier, predict_loader)
    np.savetxt(opt.output, results, fmt='%d')
    print("Output File have been written to {}".format(opt.output))

if __name__ == "__main__":
    os.system("clear")
    
    for key, value in vars(opt).items():
        print("{:15} {}".format(key, value))
    
    main()
