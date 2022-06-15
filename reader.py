"""
  FileName     [ reader.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Sample code to load an video ]
"""

import collections
import csv
import os
import pprint

import numpy as np
import pandas as pd

import skimage.transform
import skvideo.io


def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    '''
      Params:
      - video_path: video directory
      - video_category: video category (see csv files)
      - video_name: video name (unique, see csv files)
      - downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
      - rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

      Return:
      - (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)][0]
    video = os.path.join(filepath, filename)

    # --------------------------------------
    # skvideo.io.vreader()
    #   return np.ndarray
    # --------------------------------------

    videogen = skvideo.io.vreader(video)
    
    frames = []
    for frameIdx, frame in enumerate(videogen, 0):
        if frameIdx % downsample_factor == 0:
    #       frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True, multichannel=True, anti_aliasing=True).astype(np.uint8)
            frames.append(frame)
    #     else:
    #         continue

    # keep   = np.arange(0, len(videogen), downsample_factor)
    # frames = videogen[keep]

    return np.array(frames).astype(np.uint8)

def readShortFeature(video_path, video_category, video_name, downsample_factor=12):
    '''
      Params:
      - video_path: video directory
      - video_category: video category (see csv files)
      - video_name: video name (unique, see csv files)
      - downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
      - rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

      Return:
      - feature: (T, 2048) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = os.path.join(video_path, video_category)
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)][0]
    features = np.load(os.path.join(filepath, filename)).astype(np.float32)
    keep     = np.arange(0, features.shape[0], downsample_factor)
    features = features[keep]

    return features

def getVideoList(data_path):
    '''
      Params 
      - data_path: ground-truth file path (csv files)

      Return: 
      - od: Ordered dictionary of videos and labels 
            {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
      - df: The length of the video list
    '''
    result    = {}

    df = pd.read_csv(data_path)
    
    for _, row in df.iterrows():
        for column, value in row.items():
            result.setdefault(column,[]).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od, len(df)
