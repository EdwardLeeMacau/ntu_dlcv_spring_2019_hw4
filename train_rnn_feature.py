"""
  FileName     [ train_rnn_feature.py ]
  PackageName  [ HW4 ]
  Synopsis     [ RNN action recognition training methods, with preprocessed video features. ]
"""

import argparse
import datetime
import logging
import logging.config
import os
import random
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

import dataset
import utils
from classifier import Classifier
from cnn import resnet50
from rnn import LSTM_Net

# Set as true when the I/O shape of the model is fixed
# cudnn.benchmark = True
DEVICE = utils.selectDevice()

parser = argparse.ArgumentParser()
# Basic Training setting
parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--gamma", type=float, default=0.1, help="The ratio of decaying learning rate")
parser.add_argument("--milestones", type=int, nargs='*', default=[5, 35, 50, 60], help="The epoch to decay the learning rate")
parser.add_argument("--optimizer", type=str, default="SGD", help="The optimizer to use in this training")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight regularization")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--dropout", default=0.2, help="the dropout probability of the recurrent network")
parser.add_argument("--downsample", default=12, type=int, help="the downsample ratio of the training data.")
# Model dimension setting
parser.add_argument("--activation", default="LeakyReLU", help="the activation function use at training")
parser.add_argument("--layers", default=1, help="the number of the recurrent layers")
parser.add_argument("--bidirection", default=False, action="store_true", help="Use the bidirectional recurrent network")
parser.add_argument("--hidden_dim", default=128, help="the dimension of the RNN's hidden layer")
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Model parameter initialization setting
parser.add_argument("--weight_init", nargs='*', default=['orthogonal'], type=str, help="define the network weight parameter initialization methods")
parser.add_argument("--bias_init", nargs='*', default=['forget_bias_0'], type=str, help="define the network bias parameter initialization methods")
# Message logging, model saving setting
parser.add_argument("--tag", default="20190602", type=str, help="tag for this training")
parser.add_argument("--checkpoints", default="/media/disk1/EdwardLee/video/checkpoint", type=str, help="path to save the checkpoints")
parser.add_argument("--save_interval", type=int, default=1, help="interval epoch between everytime saving the model.")
parser.add_argument("--log_interval", type=int, default=100, help="interval between everytime logging the training status.")
parser.add_argument("--val_interval", type=int, default=1, help="interval between everytime validating the model performance.")
parser.add_argument("--log", default="./log", help="the root directory to save the training details")
# Devices setting
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--pretrain", type=str, help="The pretrained rnn network")
parser.add_argument("--resume", type=str, help="Path to checkpoint")
parser.add_argument("--train", default="./hw4_data/TrimmedVideos", type=str, help="path to load train datasets")
parser.add_argument("--val", default="./hw4_data/TrimmedVideos", type=str, help="path to load validation datasets")

opt = parser.parse_args()

def train(recurrent, loader, optimizer, epoch, criterion, max_trainaccs, min_trainloss):
    """ Train the classificaiton network. """
    trainloss = 0.0
    trainaccs = 0.0
    recurrent.train()
    
    for index, (feature, label, _) in enumerate(loader, 1):
        feature, label = feature.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        
        #-----------------------------------------------------------------------------
        # Setup optimizer: clean the learning rate and set the learning rate (if need)
        #-----------------------------------------------------------------------------
        # optim = utils.set_optimizer_lr(optim, lr)
        optimizer.zero_grad()

        #---------------------------------------
        # Get features, class predict:
        #   feature:       (frames, batchsize, 2048)
        #   class predict: (batchsize, num_class)
        #---------------------------------------
        predict = recurrent(feature)
        
        #---------------------------
        # Compute the loss, accuracy
        #---------------------------
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()

        predict = predict.cpu().detach().numpy()
        label   = label.cpu().detach().numpy()
        acc     = np.mean(np.argmax(predict, axis=1) == label)

        trainloss += loss.item()
        trainaccs += acc

        if index % opt.log_interval == 0:
            print("[Epoch {}] [ {:4d}/{:4d} ] [acc: {:.2%}] [loss: {:.4f}]".format(
                epoch, index, len(loader), trainaccs / index, trainloss / index
            ))

    # Print the average performance at the end time of each epoch
    trainloss = trainloss / len(loader)
    trainaccs = trainaccs / len(loader)
    print("[Epoch {}] [ {:4d}/{:4d} ] [acc: {:.2%} ({:+.2%})] [loss: {:.4f} ({:+.4f})]".format(
            epoch, len(loader), len(loader), trainaccs, trainaccs - max_trainaccs, trainloss, trainloss - min_trainloss))

    return recurrent, trainloss, trainaccs

def val(recurrent, loader, epoch, criterion, log_interval=10):
    """ Validate the recurrent network. """
    recurrent.eval()

    valaccs = 0.0
    valloss = 0.0
    
    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for index, (feature, label, _) in enumerate(loader, 1):
        feature, label = feature.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        predict = recurrent(feature)

        # loss
        loss    = criterion(predict, label)
        valloss += loss.item()
        
        # Class Accuracy
        predict = predict.cpu().detach().numpy()
        label   = label.cpu().detach().numpy()
        acc     = np.mean(np.argmax(predict, axis=1) == label)
        valaccs += acc

        if index % opt.log_interval == 0:
            print("[Epoch {}] [ {:4d}/{:4d} ]".format(epoch, index, len(loader)))

    valaccs = valaccs / len(loader)
    valloss = valloss / len(loader)

    if epoch % opt.val_interval == 0:
        print("[Epoch {}] [Validation] [ {:4d}/{:4d} ] [acc: {:.2%}] [loss: {:.4f}]".format(
                epoch, len(loader), len(loader), valaccs, valloss))

    return valaccs, valloss

def continuous_frame_recognition():
    """ Using RNN network to recognize the action. """
    start_epoch = 1

    # -----------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    # -----------------------------------------------------
    # extractor = resnet50(pretrained=True).to(DEVICE)
    recurrent = LSTM_Net(2048, opt.hidden_dim, opt.output_dim, 
                         num_layers=opt.layers, bias=True, batch_first=False, dropout=opt.dropout, 
                         bidirectional=opt.bidirection, seq_predict=False).to(DEVICE)

    # ----------------------------------------------
    # For signal direction LSTM
    #   weight_ih_l0 torch.Size([512, 2048])
    #   weight_hh_l0 torch.Size([512, 128])
    #   bias_ih_l0 torch.Size([512])
    #   bias_hh_l0 torch.Size([512])
    #
    # For bidirectional LSTM, reverse layer is added. 
    #   weight_ih_l0_reverse torch.Size([512, 2048])
    #   weight_hh_l0_reverse torch.Size([512, 128])
    #   bias_ih_l0_reverse torch.Size([512])
    #   bias_hh_l0_reverse torch.Size([512])
    # ----------------------------------------------

    # Weight_init
    if "orthogonal" in opt.weight_init:
        for layer, param in recurrent.recurrent.named_parameters():
            print("{} {}".format(layer, param.shape))
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param)

    # Bias_init
    if "forget_bias_0" in opt.bias_init:
        for layer, param in recurrent.recurrent.named_parameters():
            if layer.startswith("bias"):
                size  = param.shape[0]
                start = int(size * 0.25)
                end   = int(size * 0.5)
                param[start: end].data.fill_(0)

    if "forget_bias_1" in opt.bias_init:
        for layer, param in recurrent.recurrent.named_parameters():
            if layer.startswith("bias"):
                size  = param.shape[0]
                start = int(size * 0.25)
                end   = int(size * 0.5)
                param[start: end].data.fill_(1)

    # Set optimizer
    if opt.optimizer == "Adam":
        optimizer = optim.Adam(recurrent.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    elif opt.optimizer == "SGD":
        optimizer = optim.SGD(recurrent.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == "ASGD":
        optimizer = optim.ASGD(recurrent.parameters(), lr=opt.lr, lambd=1e-4, alpha=0.75, t0=1000000.0, weight_decay=opt.weight_decay)
    elif opt.optimizer == "Adadelta":
        optimizer = optim.Adadelta(recurrent.parameters(), lr=opt.lr, rho=0.9, eps=1e-06, weight_decay=opt.weight_decay)
    elif opt.optimizer == "Adagrad":
        optimizer = optim.Adagrad(recurrent.parameters(), lr=opt.lr, lr_decay=0, weight_decay=opt.weight_decay, initial_accumulator_value=0)
    elif opt.optimizer == "SparseAdam":
        optimizer = optim.SparseAdam(recurrent.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-08)
    elif opt.optimizer == "Adamax":
        optimizer = optim.Adamax(recurrent.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-08, weight_decay=opt.weight_dacay)
    else:
        raise argparse.ArgumentError
        
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    
    # Load parameter
    if opt.pretrain:
        recurrent = utils.loadModel(opt.pretrain, recurrent)
    if opt.resume:
        recurrent, optimizer, start_epoch, scheduler = utils.loadCheckpoint(opt.resume, recurrent, optimizer, scheduler)

    # Set criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Set dataloader
    transform = transforms.ToTensor()
    
    trainlabel   = os.path.join(opt.train, "label", "gt_train.csv")
    trainfeature = os.path.join(opt.train, "feature", "train")
    vallabel     = os.path.join(opt.val, "label", "gt_valid.csv")
    valfeature   = os.path.join(opt.val, "feature", "valid")

    train_set    = dataset.TrimmedVideos(None, trainlabel, trainfeature, downsample=opt.downsample, transform=transform)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=opt.threads)
    
    # Show the memory used by neural network
    print("The neural network allocated GPU with {:.1f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))

    #------------------
    # Train the models
    #------------------
    trainloss = []
    trainaccs = []
    valloss   = []
    valaccs   = []
    epochs    = []

    for epoch in range(start_epoch, opt.epochs + 1):
        scheduler.step()
        
        # Save the train loss and train accuracy
        max_trainaccs = max(trainaccs) if len(trainaccs) else 0
        min_trainloss = min(trainloss) if len(trainloss) else 0
        recurrent, loss, acc = train(recurrent, train_loader, optimizer, epoch, criterion, max_trainaccs, min_trainloss)
        trainloss.append(loss)
        trainaccs.append(acc)

        # validate the model with several downsample ratio
        loss_list, acc_list, label_list = [], [], []
        for downsample in [1, 2, 4, 6, 12]:
            val_set    = dataset.TrimmedVideos(None, vallabel, valfeature, downsample=downsample, transform=transform)
            val_loader = DataLoader(val_set, batch_size=1, shuffle=True, collate_fn=utils.collate_fn, num_workers=opt.threads)
            print("[Epoch {}] [Validation] [Downsample: {:2d}]".format(epoch, downsample))
            acc, loss  = val(recurrent, val_loader, epoch, criterion)
            
            loss_list.append(loss)
            acc_list.append(acc)
            label_list.append('val_{}'.format(downsample))

        valloss.append(loss_list)
        valaccs.append(acc_list)

        # Save the epochs
        epochs.append(epoch)

        # with open(os.path.join(opt.log, "problem_2", opt.tag, 'statistics.txt'), 'w') as textfile:
        #     textfile.write("\n".join(map(lambda x: str(x), (trainloss, trainaccs, valloss, valaccs, epochs))))
        
        records = list(map(lambda x: np.array(x), (trainloss, trainaccs, valloss, valaccs, epochs)))
        for record, name in zip(records, ('trainloss.txt', 'trainaccs.txt', 'valloss.txt', 'valaccs.txt', 'epochs.txt')):
            np.savetxt(os.path.join(opt.log, "problem_2", opt.tag, name), record)

        if epoch % opt.save_interval == 0:
            savepath = os.path.join(opt.checkpoints, "problem_2", opt.tag, str(epoch) + '.pth')
            utils.saveCheckpoint(savepath, recurrent, optimizer, scheduler, epoch)
        
        # Draw the accuracy / loss curve
        draw_graphs(trainloss, valloss, trainaccs, valaccs, epochs, "problem_2", label_list)
            
    return recurrent

def draw_graphs(train_loss, val_loss, train_acc, val_acc, x, problem="problem_2", label=['val_1', 'val_2', 'val_4', 'val_6', 'val_12'],
                loss_filename="loss.png", loss_log_filename="loss_log.png", acc_filename="acc.png", acc_log_filename="acc_log.png"):
    # Define the parameters
    color = plt.get_cmap('Set1')
    train_loss = np.array(train_loss, dtype=float)
    val_loss   = np.array(val_loss, dtype=float).transpose()
    train_acc  = np.array(train_acc, dtype=float)
    val_acc    = np.array(val_acc, dtype=float).transpose()
    
    # ----------------------------
    # Linear scale of loss curve
    # Log scale of loss curve
    # ----------------------------
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_loss, label="train", color='b')
    
    if len(val_loss.shape) == 2:
        for i in range(0, len(val_loss)):
            plt.plot(x, val_loss[i], label=label[i], color=color(i))
    elif len(val_loss.shape) == 1:
        plt.plot(x, val_loss, label=label[0], color=color(0))
    
    plt.plot(x, np.repeat(np.amin(val_loss), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s)")
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, loss_filename))
    
    plt.yscale('log')
    plt.title("Loss vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, loss_log_filename))

    # -------------------------------
    # Linear scale of accuracy curve
    # Log scale of accuracy curve
    # -------------------------------
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, train_acc, label="train", color='b')

    if len(val_loss.shape) == 2:
        for i in range(0, len(val_loss)):
            plt.plot(x, val_acc[i], label=label[i], color=color(i))
    elif len(val_loss.shape) == 1:
        plt.plot(x, val_loss, label=label[0], color=color(0))

    plt.plot(x, np.repeat(np.amax(val_acc), len(x)), ':')
    plt.legend(loc=0)
    plt.xlabel("Epoch(s)")
    plt.title("Accuracy vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, acc_filename))
    
    plt.yscale('log')
    plt.title("Accuracy vs Epochs")
    plt.savefig(os.path.join(opt.log, problem, opt.tag, acc_log_filename))

    plt.close('all')
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

    with open(path, "w") as textfile:
        for item, values in vars(opt).items():
            msg = "{:16} {}".format(item, values)
            print(msg)
            textfile.write(msg + '\n')

def main():
    """ Make the directory and check whether the dataset is exists """
    if not os.path.exists(opt.train):
        raise IOError("Path {} doesn't exist".format(opt.train))

    if not os.path.exists(opt.val):
        raise IOError("Path {} doesn't exist".format(opt.val))

    os.makedirs(opt.checkpoints, exist_ok=True)
    os.makedirs(os.path.join(opt.checkpoints, "problem_2"), exist_ok=True) 
    os.makedirs(os.path.join(opt.checkpoints, "problem_2", opt.tag), exist_ok=True)
    os.makedirs(opt.log, exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_2"), exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_2", opt.tag), exist_ok=True)

    # Write down the training details (opt)
    details(os.path.join(opt.log, "problem_2", opt.tag, "train_setting.txt"))

    # Train the video recognition model with single frame (cnn) method
    continuous_frame_recognition()
    
    return

if __name__ == "__main__":
    os.system("clear")    
    main()
