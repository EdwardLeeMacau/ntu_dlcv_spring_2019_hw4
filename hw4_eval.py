"""
  FileName     [ hw4_eval.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Evaluation code of hw4 ]
"""

import argparse
import os
import random
import time

import numpy as np

import reader

parser = argparse.ArgumentParser()
parser.add_argument("--gt", default="./hw4_data/TrimmedVideos/label/gt_valid.csv", type=str, help="path of the label csv file")
parser.add_argument("--pred", default="./output/p1_valid.txt", help="The predict textfile path.")
opt = parser.parse_args()

def main():
    gt, _ = reader.getVideoList(opt.gt)
    gt    = np.array(gt['Action_labels']).astype(int)
    print(gt.shape)

    predict = np.loadtxt(opt.pred, dtype=int)
    print(predict.shape)

    acc = np.mean(predict == gt)
    print("Accuracy: {:.4%}".format(acc))

if __name__ == "__main__":
    main()
