"""
  FileName     [ preprocess.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Dataset of the HW4 ]
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import dataset
import utils
from cnn import resnet50

device = utils.selectDevice()

    # -----------------------------------------------------------------
    # To save the numpy array into the file, there are several options
    #   Machine readable:
    #   - ndarray.dump(), ndarray.dumps(), pickle.dump(), pickle.dumps():
    #       Generate .pkl file.
    #   - np.save(), np.savez(), np.savez_compressed()
    #       Generate .npy file
    #   - np.savetxt()
    #       Generate .txt file.
    # -----------------------------------------------------------------

def video_to_features(data_path):
    """ Transfer the training set and validation set videos into features """

    for train in (True, False):
        datasets = dataset.TrimmedVideos(data_path, train=train, downsample=1, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

        dataloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=0)
    
        extractor = resnet50(pretrained=True).to(device).eval()

        if train:   train_val = "train"
        else:       train_val = "valid"
    
        for index, (data, _, category, name) in enumerate(dataloader, 1):
            data   = data.squeeze(0)
            datas  = np.zeros((data.shape[0], 2048), dtype=np.float)
            remain = data.shape[0]
            finish = 0

            while remain > 0:
                step = min(remain, 50)
                todo = data[finish : finish + step].to(device)
                datas[finish : finish + step] = extractor(todo).cpu().data.numpy()
                
                remain -= step
                finish += step

            print("{:4d} {:16d} {}".format(
                index, datas.shape, os.path.join(data_path, "feature", train_val, category[0], name[0] + ".npy")))

            # ------------------------------------
            # Save the feature tensor in .npy file
            # ------------------------------------
            if not os.path.exists(os.path.join(data_path, "feature", train_val, category[0])):
                os.makedirs(os.path.join(data_path, "feature", train_val, category[0]))

            np.savetxt(os.path.join(data_path, "feature", train_val, category[0], name[0] + ".npy"), datas, delimiter=',')

    return


def images_to_features(data_path):
    """ Transfer the training set and validation set images into features """

    extractor = resnet50(pretrained=True).to(device).eval()

    for folder in ("train", "valid"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        root_path  = os.path.join(data_path, "videos", folder)
        categories = os.listdir(root_path)
        
        for index, category in enumerate(categories):
            imgs_name = [name for name in os.listdir(os.path.join(root_path, category))]
            features  = np.zeros((len(imgs_name), 2048), dtype=float)

            for i, name in enumerate(imgs_name):
                img = Image.open(os.path.join(root_path, category, name))
                img = transform(img).unsqueeze(0).to(device)

                features[i] = extractor(img).squeeze(0).cpu().data.numpy()

                if i % 10 == 0:
                    print("{:4d} ".format(i))
            
            # while remain > 0:
            #     print("{:4d} / {:4d}".format(remain, remain + finish))

            #     step = min(remain, 50)
            #     todo = data[finish : finish + step].to(device)
            #     datas[finish : finish + step] = extractor(todo).cpu().data.numpy()
                
            #     remain -= step
            #     finish += step

            print("{:2d} {:16s} {}".format(
                index, str(features.shape), os.path.join(data_path, "feature", folder, category + ".npy")))

            # ------------------------------------
            # Save the feature tensor in .npy file
            # ------------------------------------
            if not os.path.exists(os.path.join(data_path, "feature", folder, category)):
                os.makedirs(os.path.join(data_path, "feature", folder, category))

            np.save(os.path.join(data_path, "feature", folder, category + ".npy"), features)

    return

if __name__ == "__main__":
    # video_to_features("./hw4_data/TrimmedVideos")
    images_to_features("./hw4_data/FullLengthVideos")
