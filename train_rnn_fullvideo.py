"""
  FileName     [ train_rnn_fullvideo.py ]
  PackageName  [ HW4 ]
  Synopsis     [ RNN action recognition training methods. ]
"""

import argparse
import datetime
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
import visualize
from classifier import Classifier
from cnn import resnet50
from rnn import LSTM_Net


# Set as true when the I/O shape of the model is fixed
# cudnn.benchmark = True
DEVICE = utils.selectDevice()

parser = argparse.ArgumentParser()
# Basic Training setting
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--gamma", type=float, default=0.1, help="The ratio of decaying learning rate")
parser.add_argument("--milestones", type=int, nargs='*', default=[], help="The epoch to decay the learning rate")
parser.add_argument("--optimizer", type=str, default="Adam", help="The optimizer to use in this training")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight regularization")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--dropout", default=0.2, help="the dropout probability of the recurrent network")
parser.add_argument("--train_downsample", default=4, type=int, help="the downsample ratio of the training data.")
parser.add_argument("--val_downsample", default=1, type=int, help="the downsample ratio of the validation data.")
# Handle very long training data strategics
parser.add_argument("--summarize", help="Remove the frames where the label is specified at the head / tail of the video.")
parser.add_argument("--sampling", default=0, type=int, help="Random Sampling the (input, label) sequence")
parser.add_argument("--k1", default=24, type=int, help="the k1 parameter of the truncated backpropagation through time, entire sequence for k1=0")
parser.add_argument("--k2", default=0, type=int, help="the k2 parameter of the truncated backpropagation through time, entire sequence for k2=0")
# Model dimension setting
parser.add_argument("--activation", default="ReLU", help="the activation function use at training")
parser.add_argument("--layers", default=1, help="the number of the recurrent layers")
parser.add_argument("--bidirection", default=False, action="store_true", help="Use the bidirectional recurrent network")
parser.add_argument("--hidden_dim", default=128, help="the dimension of the RNN's hidden layer")
parser.add_argument("--output_dim", default=11, type=int, help="the number of the class to predict")
# Model parameter initialization setting
parser.add_argument("--weight_init", nargs='*', default=['orthogonal'], type=str, help="define the network weight parameter initialization methods")
parser.add_argument("--bias_init", nargs='*', default=['forget_bias_0'], type=str, help="define the network bias parameter initialization methods")
# Message logging, model saving setting
parser.add_argument("--tag", default="20190604_bidirection_3", type=str, help="tag for this training")
parser.add_argument("--checkpoints", default="/media/disk1/EdwardLee/video/checkpoint", type=str, help="path to save the checkpoints")
parser.add_argument("--step", type=int, default=1000, help="step to test the model performance")
parser.add_argument("--save_interval", type=int, default=1, help="interval epoch between everytime saving the model.")
parser.add_argument("--log_interval", type=int, default=1, help="interval between everytime logging the training status.")
parser.add_argument("--val_interval", type=int, default=1, help="interval between everytime validating the model performance.")
parser.add_argument("--visual_interval", type=int, default=10, help="interval of epochs to visualize the prediction")
parser.add_argument("--log", default="./log", help="the root directory to save the training details")
# Devices setting
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
# Load dataset, pretrain model setting
parser.add_argument("--resume", type=str, help="Path to checkpoint (trained in Problem 3)")
parser.add_argument("--pretrain", type=str, help="The path to read the pretrained rnn network trained in Problem 2")
parser.add_argument("--train", default="./hw4_data/FullLengthVideos", type=str, help="path to load train datasets")
parser.add_argument("--val", default="./hw4_data/FullLengthVideos", type=str, help="path to load validation datasets")

opt = parser.parse_args()

def train(recurrent, loader, optimizer, epoch, criterion, max_trainaccs, min_trainloss):
    """ Train the recurrent network. """
    recurrent.train()
    
    trainloss = 0.0
    trainaccs = 0.0
    postaccs  = 0.0
    count_0s  = 0
    total_len = 0
    
    for index, (feature, label, seq_len, category, _) in enumerate(loader, 1):
        batchsize = len(seq_len)
        # print(seq_len)
        # hidden_state = torch.randn(opt.layers, batchsize, opt.hidden_dim).to(DEVICE)
        # cell_state   = torch.randn(opt.layers, batchsize, opt.hidden_dim).to(DEVICE)
        
        feature, label = feature.to(DEVICE), label.to(DEVICE)
        #-----------------------------------------------------------------------------
        # Setup optimizer: clean the learning rate and set the learning rate (if need)
        #-----------------------------------------------------------------------------
        # optim = utils.set_optimizer_lr(optim, lr)
        optimizer.zero_grad()

        #-----------------------------------------------
        # Get features, class predict:
        #   feature:       (frames, batchsize, 2048)
        #   frame_predict: (frames, batchsize, num_class)
        #------------------------------------------------
        # Execute the TPBTT technique
        if opt.k1:
            for end in range(opt.k1, seq_len[0], opt.k1):
                if end < opt.k2: continue
                
                tbptt   = [min(end, seq) for seq in seq_len]
                truncate_feature = pack_padded_sequence(feature[:end], tbptt, batch_first=False)
                truncate_label   = label[:end]
                predict, (h, c) = recurrent(truncate_feature)

                #---------------------------------------
                # Compute the loss, accuracy one-by-one
                #---------------------------------------
                predict, truncate_label = predict.view(-1, opt.output_dim), truncate_label.view(-1)
                # print("Predict.shape: {}".format(predict.shape))
                # print("Label.shape: {}".format(label.shape))

                loss = criterion(predict[-opt.k2:], truncate_label[-opt.k2:])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Predict and backpropagate the full sequence.
        feature = pack_padded_sequence(feature, seq_len, batch_first=False)
        predict, _ = recurrent(feature)
        predict, label = predict.view(-1, opt.output_dim), label.view(-1)

        # if index % 3:
        case_mask  = (label != 0)
        other_mask = (label == 0)
        loss = criterion(predict[case_mask], label[case_mask]) + 0.5 * criterion(predict[other_mask], label[other_mask])
        # else:
        # loss = criterion(predict, label)
        
        loss.backward()
        optimizer.step()

        # calculate trainloss
        trainloss += loss.item()

        # Basic prediction
        predict   = predict.cpu().detach().numpy()
        label     = label.cpu().detach().numpy()
        acc       = np.mean(np.argmax(predict, axis=1) == label)
        
        count_0   = np.sum(label == 0)
        count_0s  += count_0
        
        # Filter with the post process
        post_pred = np.argmax(predict[:, 1:], axis=1) + 1
        post_acc  = np.mean(post_pred == label)
        # post_pred = visualize.post_process(np.argmax(predict, axis=1))
        # post_acc  = np.mean(post_pred == label)

        total_len += sum(seq_len)
        trainaccs += acc * sum(seq_len)
        postaccs  += post_acc * sum(seq_len)

        # for i in range(batchsize):
        #     predict_i = predict[0: seq_len[i], i]
        #     label_i   = label[0: seq_len[i], i]
        #     acc_i     = np.mean(np.argmax(predict_i, axis=2) == label_i)
        #     post_pred_i = visualize.post_process(np.argmax(predict_i, axis=2))
        #     post_acc_i  = np.mean(np.argmax(post_pred_i, axis=2) == label_i)
        #     trainaccs += acc_i
        #     postaccs  += post_acc_i

        #-------------------------------
        # Print out the training message
        #-------------------------------
        if epoch % opt.visual_interval == 0:
            for i in range(batchsize):
                savepath = os.path.join(opt.log, "problem_3", opt.tag, "visualize", str(epoch), "train_" + category[i] + ".png")
                img_path = os.path.join(opt.train, "videos", "train", category[i])
                visualize.visualization(savepath, img_path, np.argmax(predict, axis=1), post_pred, label, sample=5, bar_height=20)

    # Print the average performance at the end time of each epoch
    trainloss = trainloss / len(loader.dataset)
    trainaccs = trainaccs / total_len
    postaccs  = postaccs / total_len
    print("[Epoch {}] [ {:4d}/{:4d} ] [acc: {:.2%} -> {:.2%} ] [0: {:.2%} {:4d}] [loss: {:.4f}]".format(
            epoch, len(loader), len(loader), trainaccs, postaccs, count_0s / total_len, count_0s, trainloss))

    return recurrent, trainaccs, trainloss

def val(recurrent: nn.Module, loader: DataLoader, epoch, criterion: nn.Module, log_interval=10, visual=True):
    """ Validate the recurrent network. """
    recurrent.eval()

    valaccs, valpostaccs, valloss = [], [], []
    count_0s  = 0
    total_len = 0
    seq_lens  = []
    
    #-------------------------------------------------------
    # Calculate the accuracy, loss one-by-one(batchsize = 1)
    #-------------------------------------------------------
    with torch.no_grad():
        for _, (feature, label, seq_len, category, _) in enumerate(loader, 1):
            feature = pack_padded_sequence(feature, seq_len, batch_first=False)
            label   = pack_padded_sequence(label, seq_len, batch_first=False)
            
            feature, label = feature.to(DEVICE), label.to(DEVICE)
            # hidden_state = torch.randn(opt.layers, opt.batch_size, opt.hidden_dim).to(DEVICE)
            # cell_state   = torch.randn(opt.layers, opt.batch_size, opt.hidden_dim).to(DEVICE)

            # predict, _ = recurrent(feature, hidden_state, cell_state)
            predict, _ = recurrent(feature)
            label, _   = pad_packed_sequence(label, batch_first=False, padding_value=0)

            predict, label = predict.view(-1, opt.output_dim), label.view(-1)
            # print("Predict.shape: {}".format(predict.shape))
            # print("Label.shape: {}".format(label.shape))

            #---------------------------
            # Compute the loss, accuracy
            #---------------------------
            loss = criterion(predict, label)
            valloss.append(loss.item())
            
            predict = predict.cpu().detach().numpy()
            label   = label.cpu().detach().numpy()
            acc     = np.mean(np.argmax(predict, axis=1) == label)
            
            count_0   = np.sum(np.argmax(predict, axis=1) == 0)
            count_0s += count_0
            
            # post_pred = visualize.post_process(np.argmax(predict, axis=1))
            post_pred = np.argmax(predict[:, 1:], axis=1) + 1
            post_acc  = np.mean(post_pred == label)
            
            total_len += seq_len[0]
            seq_lens.append(seq_len[0])
            valaccs.append(acc)
            valpostaccs.append(post_acc)

            # if epoch % 1 == 0:
            #     print("[Epoch {}] [Validation {}] [ {:4d}/{:4d} ] [acc: {:.2%} -> {:.2%}] [0: {:.2%}] [loss: {:.4f}]".format(
            #         epoch, index, len(loader), len(loader), acc, post_acc, count_0 / seq_len[0], loss.item()))

            if (epoch % opt.visual_interval == 0) and visual:
                savepath = os.path.join(opt.log, "problem_3", opt.tag, "visualize", str(epoch), "valid_" + category[0] + ".png")
                img_path = os.path.join(opt.val, "videos", "valid", category[0])
                visualize.visualization(savepath, img_path, np.argmax(predict, axis=1), post_pred, label, sample=5, bar_height=20)

    loss = sum(valloss) / len(loader.dataset)
    acc  = sum([acc * length for acc, length in zip(valaccs, seq_lens)]) / total_len
    post_acc = sum([acc * length for acc, length in zip(valpostaccs, seq_lens)]) / total_len
    
    print("[Epoch {}] [Validation  ] [ {:4d}/{:4d} ] [acc: {:.2%} -> {:.2%}] [0: {:.2%}] [loss: {:.4f}]".format(
           epoch, len(loader), len(loader), acc , post_acc, count_0s / total_len, loss))

    return valaccs, valloss

def temporal_action_segmentation():
    """ Using RNN network to segmentation the action. """
    start_epoch = 1

    #------------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    #------------------------------------------------------
    recurrent = LSTM_Net(2048, opt.hidden_dim, opt.output_dim, 
                         num_layers=opt.layers, bias=True, batch_first=False, dropout=opt.dropout, 
                         bidirectional=opt.bidirection, seq_predict=True).to(DEVICE)

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
                start = int(param.shape[0] * 0.25)
                end   = int(param.shape[0] * 0.5)
                param[start: end].data.fill_(0)

    if "forget_bias_1" in opt.bias_init:
        for layer, param in recurrent.recurrent.named_parameters():
            if layer.startswith("bias"):
                start = int(param.shape[0] * 0.25)
                end   = int(param.shape[0] * 0.5)
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
        print("Loaded pretrain model: {}".format(opt.pretrain))
    if opt.resume:
        recurrent, optimizer, start_epoch, scheduler = utils.loadCheckpoint(opt.resume, recurrent, optimizer, scheduler)
        print("Resume training: {}".format(opt.resume))

    # Set criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Set dataloader
    transform = transforms.ToTensor()

    trainlabel   = os.path.join(opt.train, "labels", "train")
    trainfeature = os.path.join(opt.train, "feature", "train")    
    vallabel     = os.path.join(opt.val, "labels", "valid")
    valfeature   = os.path.join(opt.val, "feature", "valid")
    
    train_set    = dataset.FullLengthVideos(
                        None, trainlabel, trainfeature, 
                        downsample=opt.train_downsample, 
                        transform=transform, 
                        summarize=opt.summarize, 
                        sampling=opt.sampling, 
                    )
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, collate_fn=utils.collate_fn_seq, num_workers=opt.threads)
    val_set      = dataset.FullLengthVideos(
                        None, vallabel, valfeature, 
                        downsample=opt.val_downsample, 
                        transform=transform, 
                        summarize=None, 
                        sampling=0,
                    )
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=utils.collate_fn_seq, num_workers=opt.threads)
    val_set_2    = dataset.FullLengthVideos(
                        None, vallabel, valfeature, 
                        downsample=opt.train_downsample, 
                        transform=transform, 
                        summarize=None, 
                        sampling=0
                    )
    val_loader_2 = DataLoader(val_set_2, batch_size=1, shuffle=False, collate_fn=utils.collate_fn_seq, num_workers=opt.threads)

    # Show the memory used by neural network
    print("The neural network allocated GPU with {:.1f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))

    #------------------
    # Train the models
    #------------------
    trainloss, trainaccs, valloss, valaccs = [], [], [], []
    epochs = []
    categories = [name.split('.')[0] for name in os.listdir(valfeature)]

    # Pre-test of the pretrain model
    acc, loss = val(recurrent, val_loader, 0, criterion)
    valloss.append(loss)
    valaccs.append(acc)
    epochs.append(0)

    for epoch in range(start_epoch, opt.epochs + 1):
        scheduler.step()
        
        # Save the train loss and train accuracy
        max_trainaccs = max(trainaccs) if len(trainaccs) > 0 else 0
        min_trainloss = min(trainloss) if len(trainloss) > 0 else 0
        recurrent, acc, loss = train(recurrent, train_loader, optimizer, epoch, criterion, max_trainaccs, min_trainloss)
        trainloss.append(loss)
        trainaccs.append(acc)

        # validate the model with several downsample ratio
        acc, loss = val(recurrent, val_loader, epoch, criterion)
        valloss.append(loss)
        valaccs.append(acc)

        acc, loss = val(recurrent, val_loader_2, epoch, criterion, visual=False)

        # Save the epochs
        epochs.append(epoch)

        for x, y in ((trainloss, "trainloss.txt"), (trainaccs, "trainaccs.txt"), (valloss, "valloss.txt"), (valaccs, "valaccs.txt"), (epochs, "epochs.txt")):
            np.savetxt(os.path.join(opt.log, "problem_3", opt.tag, y), np.array(x))
        
        if epoch % opt.save_interval == 0:
            savepath = os.path.join(opt.checkpoints, "problem_3", opt.tag, str(epoch) + '.pth')
            utils.saveCheckpoint(savepath, recurrent, optimizer, scheduler, epoch)
        
        # Draw the accuracy / loss curve
        draw_graphs(trainloss, valloss, trainaccs, valaccs, epochs, label=categories)
            
    return recurrent

def draw_graphs(train_loss, val_loss, train_acc, val_acc, x, problem="problem_3", label=[],
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
    plt.plot(x[1:], train_loss, label="train", color='b')
    
    if len(val_loss.shape) == 2:
        for i in range(0, len(val_loss)):
            plt.plot(x, val_loss[i], label=label[i], color=color(i))
    elif len(val_loss.shape) == 1:
        plt.plot(x, val_loss, label=label[0], color=color[0])
    
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
    plt.plot(x[1:], train_acc, label="train", color='b')

    if len(val_loss.shape) == 2:
        for i in range(0, len(val_loss)):
            plt.plot(x, val_acc[i], label=label[i], color=color(i))
    elif len(val_loss.shape) == 1:
        plt.plot(x, val_loss, label=label[0], color=color[0])

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
    os.makedirs(os.path.join(opt.checkpoints, "problem_3"), exist_ok=True) 
    os.makedirs(os.path.join(opt.checkpoints, "problem_3", opt.tag), exist_ok=True)
    os.makedirs(opt.log, exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_3"), exist_ok=True)
    os.makedirs(os.path.join(opt.log, "problem_3", opt.tag), exist_ok=True)

    # Write down the training details (opt)
    details(os.path.join(opt.log, "problem_3", opt.tag, "train_setting.txt"))

    # Train the video recognition model with single frame (cnn) method
    temporal_action_segmentation()
    
    return

if __name__ == "__main__":
    os.system("clear")
    main()
