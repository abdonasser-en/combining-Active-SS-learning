
from strategy_utils_framework.strategy import Strategy
import numpy as np
import random
from sklearn import preprocessing
from torch import nn
import sys, os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from strategy_utils_framework.utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import time
from torchvision.utils import save_image

from tqdm import tqdm
from strategy_utils_framework.util import get_unique_folder
from sklearn.metrics import pairwise_distances
from torchmetrics import MatthewsCorrCoef
from torchvision import transforms
import pathlib

class Framework2(Strategy):
    def __init__(self,X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(Framework2, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
    

    def predict_coefficient_w(self,X,Y,q_idx):
        transform=transforms.ToTensor()
        loader_te = DataLoader(self.handler(X[q_idx], Y[q_idx], transform=transform), pin_memory=True, 
                            shuffle=False, **self.args.loader_te_args)
        
        self.net.eval()
        predictions = []
        with torch.no_grad():
            for x, _,_ in loader_te:
                # Move the batch to the appropriate device (CPU or GPU)
                x= x.to(self.device)  # device could be 'cuda' or 'cpu'
                # Forward pass to obtain predictions
                batch_predictions,_= self.net(x)
                
                # Append batch predictions to the list
                predictions.append(batch_predictions.cpu().numpy())  # Move predictions to CPU and convert to numpy

        # Concatenate predictions from all batches
        predictions = np.concatenate(predictions, axis=0)
        return np.argmax(predictions,axis=1)
    
    
    def predict_coefficient(self,X,Y,q_idx):
        transform=self.args.transform_te 
        loader_te = DataLoader(self.handler(X[q_idx], Y[q_idx], transform=transform), pin_memory=True, 
                            shuffle=False, **self.args.loader_te_args)
        
        self.net.eval()
        predictions = []
        with torch.no_grad():
            for x, _,_ in loader_te:
                # Move the batch to the appropriate device (CPU or GPU)
                x= x.to(self.device)  # device could be 'cuda' or 'cpu'
                # Forward pass to obtain predictions
                batch_predictions,_= self.net(x)
                
                # Append batch predictions to the list
                predictions.append(batch_predictions.cpu().numpy())  # Move predictions to CPU and convert to numpy

        # Concatenate predictions from all batches
        predictions = np.concatenate(predictions, axis=0)
        return np.argmax(predictions,axis=1)
    

    transforms.ToTensor()