from joblib.externals.cloudpickle.cloudpickle import instance
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
from .utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import time
from torchvision.utils import save_image

from tqdm import tqdm
from strategy_utils_framework.util import get_unique_folder
from sklearn.metrics import pairwise_distances
from torchmetrics import MatthewsCorrCoef
import pathlib

class Strategy:
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        self.X = X  # vector
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te

        self.idxs_lb = idxs_lb # bool type
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = net.to(self.device)
        

        # for reproducibility
        self.g = torch.Generator()
        self.g.manual_seed(0)

    def seed_worker(self, worker_id):
        """
        To preserve reproducibility when num_workers > 1
        """
        # https://pytorch.org/docs/stable/notes/randomness.html
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.net.train()

        accFinal = 0.
        train_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device) 
            nan_mask = torch.isnan(x)
            if nan_mask.any():
                raise RuntimeError(f"Found NAN in input indices: ", nan_mask.nonzero())

            # exit()
            optimizer.zero_grad()

            out,e1 = self.net(x)
            nan_mask_out = torch.isnan(y)
            if nan_mask_out.any():
                raise RuntimeError(f"Found NAN in output indices: ", nan_mask.nonzero())
                
            loss = F.cross_entropy(out, y)

            train_loss += loss.item()
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            
            loss.backward()
            
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.net.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
            
            if batch_idx % 10 == 0:
                print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return accFinal / len(loader_tr.dataset.X), train_loss

    
    def train(self, alpha=0.1, n_epoch=5):
        # self.clf =  deepcopy(self.net)
        # if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.clf = nn.parallel.DistributedDataParallel(self.clf,
                                                        # find_unused_parameters=True,
                                                        # )
        #nn.dataParallel
        # self.clf = self.clf.to(self.device)
        parameters = self.net.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.
        best_test_acc = 0.
        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr

            train_data = self.handler(self.X[idxs_train], 
                                torch.Tensor(self.Y[idxs_train]).long() if type(self.Y) is np.ndarray else  torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform)

            loader_tr = DataLoader(train_data, 
                                    shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args.loader_tr_args)
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{}({}+{}) Need: {:02d}:{:02d}:{:02d}]'.format(self.args.framework,self.args.ALstrat,self.args.SSLstrat, need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr, optimizer)
                test_acc = self.predict(self.X_te, self.Y_te)
                # measure elapsed time
                epoch_time.update(time.time() - ts)
                print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Test Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               1. - recorder.max_accuracy(False)))
                recorder.update(epoch, train_los, train_acc, 0, test_acc)

                if self.args.save_model and test_acc > best_test_acc:
                    best_test_acc = test_acc
                    self.save_model()
            recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            # self.clf = self.clf.module

        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc                


    def predict(self, X, Y):
        transform=self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)
        
        self.net.eval()

        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out,e1= self.net(x)
                pred = out.max(1)[1]                
                correct +=  (y == pred).sum().item() 

            test_acc = 1. * correct / len(Y)
   
        return test_acc

    def get_prediction(self, X, Y):
        transform=self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)

        P = torch.zeros(len(X)).long().to(self.device)

        self.net.eval()


        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out,e1= self.net(x)
                pred = out.max(1)[1]     
                P[idxs] = pred           
                correct +=  (y == pred).sum().item() 
   
        return P

    def predict_prob(self, X, Y):
        transform = self.args.transform_te 
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, pin_memory=True, **self.args.loader_te_args)

        self.net.eval()

        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out,e1 = self.net(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    # def predict_prob_dropout(self, X, Y, n_drop):
    #     transform = self.args.transform_te 
    #     loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
    #                         shuffle=False, **self.args.loader_te_args)

    #     self.net.train()

    #     probs = torch.zeros([len(Y), len(np.unique(Y))])
    #     with torch.no_grad():
    #         for i in range(n_drop):
    #             print('n_drop {}/{}'.format(i+1, n_drop))
    #             for x, y, idxs in loader_te:
    #                 x, y = x.to(self.device), y.to(self.device) 
    #                 out,e1= self.net(x)
    #                 prob = F.softmax(out, dim=1)
    #                 probs[idxs] += prob.cpu().data
    #     probs /= n_drop
        
    #     return probs

    # def predict_prob_dropout_split(self, X, Y, n_drop):
    #     transform = self.args.transform_te
    #     loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
    #                         shuffle=False, **self.args.loader_te_args)

    #     self.net.train()

    #     probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
    #     with torch.no_grad():
    #         for i in range(n_drop):
    #             print('n_drop {}/{}'.format(i+1, n_drop))
    #             for x, y, idxs in loader_te:
    #                 x, y = x.to(self.device), y.to(self.device) 
    #                 out,e1 = self.net(x)
    #                 probs[i][idxs] += F.softmax(out, dim=1).cpu().data
    #         return probs

    
    def save_model(self):
        # save model and selected index
        save_path = os.path.join(self.args.save_path,self.args.dataset+'_checkpoint')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = '%.1f'%float(100*labeled/len(self.X))
        torch.save(self.clf, os.path.join(save_path, self.args.framework+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.pkl'))
        print('save to ',os.path.join(save_path, self.args.framework+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.pkl'))
        path = os.path.join(save_path, self.args.framework+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.npy')
        np.save(path,self.idxs_lb)

    def load_model(self):
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = '%.1f'%float(100*labeled/len(self.X))
        save_path = os.path.join(self.args.save_path,self.args.dataset+'_checkpoint')
        self.clf = torch.load(os.path.join(save_path, self.args.framework+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.pkl'))
        self.idxs_lb = np.load(os.path.join(save_path, self.args.framework+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.seed)+'.npy'))

  

