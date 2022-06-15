"""
  FileName     [ data_augment.py ]
  PackageName  [ HW4 ]
  Synopsis     [ To generate difference setting dataset for Problem 3 ]
"""

import argparse
import collections
import datetime
import itertools
import os
import random

import numpy as np
import torch
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
import visualize
from tqdm import tqdm

# Set as true when the I/O shape of the model is fixed

parser = argparse.ArgumentParser()
# Devices setting
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--train", default="./hw4_data/FullLengthVideos", type=str, help="path to load train datasets")
parser.add_argument("--augment", default="./hw4_data/FullLengthVideosAugment", type=str, help="path to save augment datasets")
parser.add_argument("--save", default=True, action="store_true")
opt = parser.parse_args()

def augmentation_problem3():
    label_path   = os.path.join(opt.train, "labels", "train")
    feature_path = os.path.join(opt.train, "feature", "train")
    augment_label_path   = os.path.join(opt.augment, "labels", "train")
    augment_feature_path = os.path.join(opt.augment, "feature", "train")
    
    augment_dataset = dataset.FullLengthVideos(
        video_path=None, 
        label_path=label_path, 
        feature_path=feature_path, 
        downsample=1, 
        rescale=1, 
        transform=None, 
        summarize=None, 
        sampling=False, 
        truncate=(0, 0)
    )
    augment_loader = DataLoader(augment_dataset, batch_size=1, shuffle=False, num_workers=0)

    for _, (frames, labels, category) in enumerate(augment_loader, 1):
        category = category[0]
        print("Category: {:32s}".format(category))
        
        frames = frames.squeeze(0).cpu().detach().data.numpy()
        labels = labels.cpu().detach().data.numpy().transpose()
        print("- Frames: {}".format(frames.shape))
        print("- Labels: {}".format(labels.shape))
        print()

        # statis = summarize(labels.tolist())
        # for s in statis:
        #     print(s)
        
        # functions = [segmentation, segmentation_2]
        functions = [segmentation_2]
        for function in functions:
            print(function)
            batch_frames, batch_labels, marks = function(frames, labels)
            print("- Frames: {}".format([frame.shape for frame in batch_frames]))
            print("- Labels: {}".format([label.shape for label in batch_labels]))
            print("- Marks:  {}".format(marks))

            if opt.save:
                # Case 1: Only generate 1 file from 1 full video
                if marks is None:
                    features_to = os.path.join(augment_feature_path, category + ".npy")
                    label_to    = os.path.join(augment_label_path, category + ".txt")
                    np.save(features_to, batch_frames[0])
                    np.savetxt(label_to, batch_labels[0], fmt='%d')

                # Case 2: generates multi-video segments from 1 full video
                if marks is not None:
                    for index in range(len(marks)):
                        start, end  = marks[index]
                        features_to = os.path.join(augment_feature_path, category + "_" + str(start) + "_" + str(end) + ".npy")
                        label_to    = os.path.join(augment_label_path, category + "_" + str(start) + "_" + str(end) + ".txt")

                        np.save(features_to, batch_frames[index])
                        np.savetxt(label_to, batch_labels[index])

        print("="*32)
    
    if opt.save:
        print("Augment data saved")

    return

def details(path):
    makedirs = []
    
    folder = os.path.dirname(path)
    while not os.path.exists(folder):
        makedirs.append(folder)
        folder = os.path.dirname(folder)

    while len(makedirs) > 0:
        makedirs, folder = makedirs[:-1], makedirs[-1]
        os.makedirs(folder)

def removeAllWithLabel(frames: np.array, labels: np.array, target=0) -> tuple:
    keep = (labels != target).flatten()
    return [frames[keep]], [labels[keep]], None

def segmentation(frames: np.array, labels: np.array, target=0, frames_limit=5) -> tuple:
    batch_frames, batch_labels, marks = [], [], []
    
    timemarks  = visualize.convert_marks(labels)
    start, end = 0, 0
    
    # ------------------------------
    # Algorithm description:
    #   Tag<Start> >= Tag<End>:
    #     -> Finding the startpoint
    #   Tag<Start> < Tag<End>
    #     -> Finding the endpoint
    # ------------------------------
    for _, length, k in timemarks:
        if k == target:
            if start >= end:
                start += length
            elif length > frames_limit:
                marks.append((start, end + frames_limit))
                start = end + length
                end   = end + length
            else:
                end += length

        # Not found the target
        if k != target:
            if start <= end:
                end += length
                continue

            if start > end:
                end = start + length
                continue

    for start, end in marks:
        batch_frames.append(frames[start: end])
        batch_labels.append(labels[start: end])

    return batch_frames, batch_labels, marks

def segmentation_2(frames: np.array, labels: np.array, target=0, frames_limit=8) -> tuple:
    batch_frames, batch_labels, marks = [], [], []
    
    timemarks  = visualize.convert_marks(labels)
    start, end = 0, 0
    
    # ------------------------------
    # Algorithm description:
    #   Tag<Start> >= Tag<End>:
    #     -> Finding the startpoint
    #   Tag<Start> < Tag<End>
    #     -> Finding the endpoint
    # ------------------------------
    for _, length, k in timemarks:
        # If FOUND the target
        if k == target:
            if length >= frames_limit:
                if (end > start): marks.append((start, end))
                start = end + length - frames_limit
                end   = end + length
            elif length < frames_limit:
                end += length

        # If NOT found the target
        if k != target:
            end += length

    for start, end in marks:
        batch_frames.append(frames[start: end])
        batch_labels.append(labels[start: end])

    return batch_frames, batch_labels, marks

def summarize(label) -> tuple:
    marks = visualize.convert_marks(label)
    
    counter = collections.Counter()
    for _, length, k in marks:
        counter[k[0]] += length
    
    return sorted(counter.most_common(11), key=lambda x: x[0])

def main():
    # Make the directory and check whether the dataset is exists
    if not os.path.exists(opt.train):
        raise IOError("Path {} doesn't exist".format(opt.train))

    if not os.path.exists(opt.augment):
        os.makedirs(opt.augment, exist_ok=True)

    for folder in ["feature", "labels", "videos"]:
        for train_val in ("train", "valid"):
            if not os.path.exists(os.path.join(opt.augment, folder)):
                os.makedirs(os.path.join(opt.augment, folder))

            if not os.path.exists(os.path.join(opt.augment, folder, train_val)):
                os.makedirs(os.path.join(opt.augment, folder, train_val))
    
    # Train the video recognition model with single frame (cnn) method
    augmentation_problem3()
    
    return

if __name__ == "__main__":
    os.system("clear")
    main()
