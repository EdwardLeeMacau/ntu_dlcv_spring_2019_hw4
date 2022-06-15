"""
  FileName     [ utils.py ]
  PackageName  [ HW4 ]
  Synopsis     [ Utility functions in package HW4 ]
"""

import sys

import numpy as np
import torch
from torch import nn, optim
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

trimmedVideos_feature    = []
fullLengthVideos_feature = []

def cosine_annealing_with_warm_restart(iteration, T_max, eta_min, last_epoch, factor):
    return

def set_optimizer_lr(optimizer, lr):
    """ set the learning rate in an optimizer, without rebuilding the whole optimizer """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def collate_fn(batch):
    """
      To define a function that reads the video by batch.
 
      Params:
      - batch: 
          In pytorch, dataloader generate batch of traindata by this way:
            `self.collate_fn([self.dataset[i] for i in indices])`
          
          In here, batch means `[self.dataset[i] for i in indices]`
          It's a list contains (datas, labels)

      Return:
      - batch: the input tensors in form of PackSequence()
    """
    # ---------------------------------
    # batch[i][j]
    #   the type of batch[i] is tuple
    # 
    #   i=(0, size) means the batchsize
    #   j=(0, 1) means the data / label
    # ---------------------------------
    
    # Sorted the batch with the video length with the descending order
    batch   = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    seq_len = [x[0].shape[0] for x in batch]

    label = None
    if batch[0][1] is not None:
        label = torch.cat([x[1].unsqueeze(0) for x in batch], dim=0)
    
    batch = pack_padded_sequence(pad_sequence([x[0] for x in batch], batch_first=False), seq_len, batch_first=False)
    
    return (batch, label, seq_len)

def collate_fn_valid(batch):
    """
      The collate_fn function used in valid code. 
      
      Main process:
        Return the frames and seq_len 
    """
    batch   = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    seq_len = [x[0].shape[0] for x in batch]

    label = None
    if batch[0][1] is not None:
        label = pad_sequence([x[1] for x in batch], batch_first=False)
    
    batch      = torch.cat([x[0] for x in batch], dim=0)

    return (batch, label, seq_len)


def collate_fn_seq(batch):
    batch   = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    seq_len = [x[0].shape[0] for x in batch]

    label      = pad_sequence([x[1] for x in batch], batch_first=False)
    raw_length = [x[3] for x in batch]
    categories = [x[2] for x in batch]
    batch      = pad_sequence([x[0] for x in batch], batch_first=False)
    
    return (batch, label, seq_len, categories, raw_length)

def selectDevice():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def saveCheckpoint(checkpoint_path, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR, epoch, feature=None):
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict()
    }

    if feature:
        state['feature'] = feature.state_dict()

    torch.save(state, checkpoint_path)

    return

def loadCheckpoint(checkpoint_path: str, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR, feature=None):
    state = torch.load(checkpoint_path)

    resume_epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    if feature:
        feature.load_state_dict(state['feature'])
        return feature, model, optimizer, resume_epoch, scheduler

    return model, optimizer, resume_epoch, scheduler

def saveModel(checkpoint_path: str, model: nn.Module, feature=None):
    """
      Params:
      - checkpoint_path: the directory of the model parameter
      - feature: the structure of the feature extractor
      - model: If cnn -> classifier
               If rnn -> recurrent
      - pretrained: If True, ignore the feature extractor pretrained model
                    If False, save the model parameter to the file
    """
    state = {'state_dict': model.state_dict()}

    if feature:
        state['feature'] = feature.state_dict()

    torch.save(state, checkpoint_path)

def loadModel(checkpoint_path: str, model: nn.Module, feature=None):
    """
      Params:
      - checkpoint_path: the directory of the model parameter
      - feature: the structure of the feature extractor
      - model: If cnn -> classifier
               If rnn -> recurrent
      - pretrained: If True, load the feature extractor pretrained model
                    If False, load the model parameter from the saved file
    """
    state = torch.load(checkpoint_path)
    
    model.load_state_dict(state['state_dict'])
    
    if feature:
        feature.load_state_dict(state['feature'])
        return feature, model

    return model

def checkpointToModel(checkpoint_path: str, model_path: str):
    state = torch.load(checkpoint_path)

    newState = {
        'state_dict': state['state_dict']
    }

    torch.save(newState, model_path)
