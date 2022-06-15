"""
  FileName     [ dataset.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Dataset of the HW4 ]

  - Dataset:
    TrimmedVideos:              Prob 1, 2
    TrimmedVideosPredict:       Prob 1, 2
    FullLengthVideos:           Prob 3
    FullLengthVideosPredict:    Prob 3
"""

import csv
import itertools
import os
import pprint
import random
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import reader
import utils
from cnn import resnet50


class TrimmedVideos(Dataset):
    def __init__(self, video_path, label_path, feature_path, downsample=1, rescale=1, sample=None, transform=None):
        assert ((video_path is not None) or (feature_path is not None)), "Video_path or feature_path is needed for Dataset: FullLenghtVideos"
        
        self.label_path   = label_path
        self.video_path   = video_path
        self.feature_path = feature_path
        self.downsample = downsample
        self.rescale    = rescale
        self.sample     = sample
        self.transform  = transform
 
        self.video_list, self.len = reader.getVideoList(self.label_path)
        
        # else:
        #     categories = os.listdir(self.video_path)
        #     self.video_list = [name for name in os.path.join(self.video_path, category) for category in categories]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video_name     = self.video_list['Video_name'][index]
        video_category = self.video_list['Video_category'][index]
        
        video_label    = None
        if 'Action_labels' in self.video_list:
            video_label = torch.LongTensor([self.video_list['Action_labels'][index]])

        # ---------------------------------------------------------------
        # Sample for HW4.1, pick the fixed number of frames 
        # Downsample for HW4.2, pick the frames with the downsampling rate
        # ----------------------------------------------------------------
        if self.feature_path is not None:
            video = reader.readShortFeature(self.feature_path, video_category, video_name, downsample_factor=self.downsample)
        elif self.video_path is not None:
            video = reader.readShortVideo(self.video_path, video_category, video_name, downsample_factor=self.downsample, rescale_factor=self.rescale)
        
        if self.sample:
            step  = video.shape[0] / self.sample
            frame = np.around(np.arange(0, video.shape[0], step), decimals=0).astype(int)
            video = video[frame]

        # ---------------------------------------------------
        # Features Output dimension:   (frames, 2048)
        # Full video Output dimension: (frames, channel, height, width)
        # ---------------------------------------------------
        if self.transform:
            if self.feature_path is not None:
                tensor = self.transform(video)
                
                return tensor.squeeze(0), video_label

            if self.video_path is not None:
                tensor = torch.zeros(video.shape[0], 3, 240, 320).type(torch.float32)
                
                for i in range(video.shape[0]):
                    tensor[i] = self.transform(video[i])
                
                return tensor, video_label

        return video, video_label

class FullLengthVideos(Dataset):
    def __init__(self, video_path, label_path, feature_path, downsample=1, rescale=1, transform=None,
                 summarize=None, sampling=0):        
        assert ((video_path is not None) or (feature_path is not None)), "Video_path or feature_path is needed for Dataset: FullLenghtVideos."
        assert ((not summarize) or (label_path is not None)), "Summarize can only be used in training mode, self.label_path shouldn't be 0."
        
        self.label_path   = label_path
        self.video_path   = video_path
        self.feature_path = feature_path
        self.downsample = downsample
        self.rescale    = rescale
        self.transform  = transform
        self.summarize  = summarize
        self.sampling   = sampling

        if video_path is not None:
            self.categories = [folder for folder in os.listdir(video_path)]
        elif feature_path is not None:
            self.categories = [filename.split('.')[0] for filename in os.listdir(feature_path)]
        elif label_path is not None:
            self.categories = [filename.split('.')[0] for filename in os.listdir(label_path)]

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, index):
        video_category = self.categories[index]
        
        # Read Labels
        video_label = None
        if self.label_path is not None:
            video_label = torch.from_numpy(np.loadtxt(os.path.join(self.label_path, video_category + '.txt'))).type(torch.LongTensor)

        # Read videos
        if self.video_path is not None:
            frames = sorted([os.path.join(self.video_path, video_category, name) for name in os.listdir(os.path.join(self.video_path, video_category))])
            frames = [Image.open(name) for name in frames]

        if self.feature_path is not None:
            frames = np.load(os.path.join(self.feature_path, video_category + ".npy"))
        
        # ----------------------------------------------------------------------------------
        # downsample: 
        # rescale:
        # summarize:  Remove the frequently appears labels (The head and tail in this case)
        # sampling:   Sampling the (input, label) sequence
        # truncate:   TBPTT technique, need to crop the (input, label)
        # ----------------------------------------------------------------------------------
        raw_length = len(frames)

        if self.downsample > 1:
            keep = np.arange(0, len(frames), self.downsample)
            frames = frames[keep]
            
            if self.label_path is not None:
                video_label = video_label[keep] 

        if self.summarize is not None:
            target = self.summarize

            start, length, mark = 0, 0, []
            for k, g in itertools.groupby(video_label):
                length = len(list(g))
                start += length
                # print(start)
                if (k == target) and (len(mark) == 0):
                    mark.append(start)
                    continue
            mark.append(start - length)

            # print("Summarize: ", mark[0], mark[1], len(frames))
            frames = frames[mark[0]: mark[1]]
            video_label = video_label[mark[0]: mark[1]]

        if self.sampling:
            raw_length = len(frames)

            length = min(self.sampling, len(frames))
            start  = random.randint(0, len(frames) - length)

            frames = frames[start: start + length]
            video_label = video_label[start: start + length]
            # print("Sampling: ", 0, start, start + length, raw_length)

        # -------------------------------------------------------------
        # Features Output dimension: (frames, 2048)
        # Full video Output dimension: (frames, channel, height, width)
        # -------------------------------------------------------------
        if self.transform:
            if self.feature_path is not None:
                tensor = self.transform(frames).type(torch.float32)
                
                return tensor.squeeze(0), video_label, video_category, raw_length

            if self.video_path is not None:
                tensor = torch.zeros(len(frames), 3, 240, 320).type(torch.float32)
                for i in range(len(frames)):
                    tensor[i] = self.transform(frames[i])
            
                return tensor.squeeze(0), video_label, video_category, raw_length

        return frames, video_label, video_category, raw_length


def main():
    videopath = "./hw4_data/TrimmedVideos/video/train"
    labelpath = "./hw4_data/TrimmedVideos/label/train"
    featurepath = "./hw4_data/TrimmedVideos/feature/train"

    # video_unittest(datapath)
    # print("Video Unittest Passed!")

    read_feature_unittest(videopath, labelpath, featurepath)
    print("Read Features Unittest Passed!")

    # predict_unittest(os.path.join(datapath, "video", "valid"))
    # print("Predict Unittest Passed!")

    # datapath = "./hw4_data/FullLengthVideos"

if __name__ == "__main__":
    main()
