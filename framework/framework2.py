
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
from sklearn.metrics import cohen_kappa_score as c_k

class Framework2(Strategy):
    def __init__(self,X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        # Initialize the Framework2 instance
        # Args:
        #   X: Training data features
        #   Y: Training data labels
        #   X_te: Testing data features
        #   Y_te: Testing data labels
        #   idxs_lb: Labeled data indices
        #   net: Neural network model
        #   handler: Data handling utility
        #   args: Additional arguments and configurations
        super(Framework2, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.X_tr = X
        self.Y_tr = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.idxs_lb = idxs_lb
        self.model = net
        self.handler = handler
        self.args = args
    

    def predict_coefficient_w(self,X,Y,q_idx):
         # Predict cohen's Kappa coefficients for given data points without augmentation.
        # Args:
        #   X: Data features
        #   Y: Data labels
        #   q_idx: Indices of the data points to predict coefficients for
        # Returns:
        #   An array of predicted coefficients
        # This method loads data, evaluates the neural network, and returns predicted coefficients.
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
        # Predict cohen's Kappa coefficients for given data points using a strong transformation.
        # Args:
        #   X: Data features
        #   Y: Data labels
        #   q_idx: Indices of the data points to predict coefficients for
        # Returns:
        #   An array of predicted coefficients
        # This method transforms data, evaluates the neural network, and returns predicted coefficients.
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
                print(batch_predictions)
                # Append batch predictions to the list
                predictions.append(batch_predictions.cpu().numpy())  # Move predictions to CPU and convert to numpy

        # Concatenate predictions from all batches
        predictions = np.concatenate(predictions, axis=0)
        return np.argmax(predictions,axis=1)
    


    def train_framework(self,stratAl:object,stratSSL:object,NUM_ROUND,NUM_QUERY,alpha,n_epochs):
        # Train the framework using active learning and semi-supervised learning strategies.
        # Args:
        #   stratAl: Active learning strategy
        #   stratSSL: Semi-supervised learning strategy
        #   NUM_ROUND: Number of training rounds
        #   NUM_QUERY: Number of data points to query in each round
        #   alpha: Learning rate
        #   n_epochs: Number of training epochs
        # Returns:
        #   An array of accuracy values for each training round
        # This method trains the model, performs active and semi-supervised learning rounds,
        # and records the accuracy for each round.

        print(f' Sratgey for active learning{stratAl} and strategy for semi-supervised learning used {stratSSL}')
        stratAl=stratAl(self.X_tr, self.Y_tr, self.X_te, self.Y_te, self.idxs_lb, self.net, self.handler, self.args,self.n_pool,self.device)
        stratSSL=stratSSL(self.X_tr, self.Y_tr, self.X_te, self.Y_te, self.idxs_lb, self.net, self.handler, self.args,self.n_pool,self.device,self.predict,self.g)

        # Train father
        self.train(alpha=2e-3,n_epoch=5)


        test_acc=self.predict(self.X_te,self.Y_te)
        acc = np.zeros(NUM_ROUND+1)
        acc[0] = test_acc
        for rd in range(1, NUM_ROUND+1):

            labeled = len(np.arange(self.n_pool)[self.idxs_lb])
            if NUM_QUERY > int(self.args.nEnd*self.n_pool/100) - labeled:
                NUM_QUERY = int(self.args.nEnd*self.n_pool/100) - labeled
            
            # query
            ts = time.time()
            output = stratAl.query(NUM_QUERY)
            q_idxs = output
            print(q_idxs)
            #predict father and son
            prediction=self.predict_coefficient(self.X_tr ,self.Y_tr,q_idxs)
            prediction_w=self.predict_coefficient_w(self.X_tr, self.Y_tr,q_idxs)
            # compute the coefficient
            cof=c_k(prediction,prediction_w)
            print(cof)    
            # update
            self.idxs_lb[q_idxs] = True
            te = time.time()
            tp = te - ts
            self.update(self.idxs_lb)

            if cof<0.8:
                # Al_methods
                print( 'AL Methods')
                print('Round {}/{}'.format(rd, NUM_ROUND), flush=True)

                if hasattr(stratAl, 'train'):
                
                    best_test_acc=stratAl.train(alpha=2e-3, n_epoch=5)
                else: best_test_acc = self.train(alpha=2e-3, n_epoch=5)
                print(best_test_acc)


                t_iter = time.time() - ts
                
                # round accuracy
                # test_acc = strategy.predict(X_te, Y_te)
                acc[rd] = best_test_acc
            else:
                #SSL methods
                print("SSL Methods")
                print('Round {}/{}'.format(rd, NUM_ROUND), flush=True)
                best_test_acc = stratSSL.train(alpha=2e-3, n_epoch=5)


                t_iter = time.time() - ts
                
                # round accuracy
                # test_acc = strategy.predict(X_te, Y_te)
                acc[rd] = best_test_acc
                print(acc[rd])
        print(acc)
        return acc



    

