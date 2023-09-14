import numpy as np
import random
import sys

import os
import argparse
from preprocess.dataset import get_dataset, get_handler
from torchvision import transforms
import torch
import csv
import time
import framework
import al_methods
import ssl_methods
import models
from strategy_utils_framework.utils import print_log
# import torch.distributed as dist

os.environ['CUBLAS_WORKSPACE_CONFIG']= ':16:8'
#Framework

query_framework= sorted(name for name in framework.__dict__
                     if callable(framework.__dict__[name]))
#AL Methods 
query_strategies_al = sorted(name for name in al_methods.__dict__
                     if callable(al_methods.__dict__[name]))
#SSL Methods
query_strategies_ssl=sorted(name for name in ssl_methods.__dict__
                     if callable(ssl_methods.__dict__[name]))




###############################################################################
parser = argparse.ArgumentParser()
# strategy
parser.add_argument('--framework', help='choose framework for comibinaition active and semi-supervised learning',
                     type=str, choices=query_framework, 
                    default='rand')

parser.add_argument('--ALstrat', help='choose Al method for framework',
                     type=str, choices=query_strategies_al, 
                    default='rand')
parser.add_argument('--SSLstrat', help='choose ssl method for framework',
                     type=str, choices=query_strategies_ssl, 
                    default='rand')


parser.add_argument('--nQuery',  type=float, default=10,
                    help='number of points to query in a batch (%)')
parser.add_argument('--nStart', type=float, default=10,
                    help='number of points to start (%)')
parser.add_argument('--nEnd',type=float, default=50,
                        help = 'total number of points to query (%)')
parser.add_argument('--nEmb',  type=int, default=256,
                        help='number of embedding dims (mlp)')
parser.add_argument('--seed', type=int, default=1,
                    help='the index of the repeated experiments', )

# model and data
parser.add_argument('--model', help='model - resnet50, vgg, or mlp', type=str)
parser.add_argument('--dataset', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--data_path', help='data path', type=str, default='./datasets')
parser.add_argument('--save_path', help='result save save_dir', default='./save')
parser.add_argument('--save_file', help='result save save_dir', default='result.csv')



# for ensemble based methods
parser.add_argument('--n_ensembles', type=int, default=1, 
                    help='number of ensemble')

# for proxy based selection
parser.add_argument('--proxy_model', type=str, default=None,
                    help='the architecture of the proxy model')

# training hyperparameters
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--n_epoch', type=int, default=10,
                    help='number of training epochs in each iteration')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate. 0.01 for semi')
parser.add_argument('--gammas',
                    type=float,
                    nargs='+',
                    default=[0.1, 0.1],
                    help=
                    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--save_model', 
                    action='store_true',
                    default=False, help='save model every steps')
parser.add_argument('--load_ckpt', 
                    action='store_true',
                    help='load model from memory, True or False')


# automatically set
# parser.add_argument("--local_rank", type=int)

##########################################################################
args = parser.parse_args()
# set the backend of the distributed parallel
# ngpus = torch.cuda.device_count()
# dist.init_process_group("nccl")

############################# For reproducibility #############################################

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    # True ensures the algorithm selected by CUFA is deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    # False ensures CUDA select the same algorithm each time the application is run
    torch.backends.cudnn.benchmark = False

############################# Specify the hyperparameters #######################################
 
args_pool = {'mnist':
                { 
                 'n_class':10,
                 'channels':1,
                 'size': 28,
                 'transform_tr': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': (0.1307,), 'std': (0.3081,)},
                },

            'svhn':
                {
                 'n_class':10,
                'channels':3,
                'size': 32,
                'transform_tr': transforms.Compose([ 
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970)},
                },
            'cifar10':
                {
                 'n_class':10,
                 'channels':3,
                 'size': 32,
                 'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 512, 'num_workers': 8},
                 'normalize':{'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2470, 0.2435, 0.2616)},
                 },


            'cifar100': 
               {
                'n_class':100,
                'channels':3,
                'size': 32,
                'transform_tr': transforms.Compose([
                                transforms.RandomCrop(size = 32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
                'transform_te': transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
                'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                'loader_te_args':{'batch_size': 512, 'num_workers': 8},
                'normalize':{'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761)},
                }
        }

###############################################################################
###############################################################################

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)
    log = os.path.join(args.save_path,
                        'log_seed_{}.txt'.format(args.seed))
    # print the args
    print(args.save_model)
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(str(state), log)
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    # load the dataset specific parameters
    dataset_args = args_pool[args.dataset]
    args.n_class = dataset_args['n_class']
    args.img_size = dataset_args['size']
    args.channels = dataset_args['channels']
    args.transform_tr = dataset_args['transform_tr']
    args.transform_te = dataset_args['transform_te']
    args.loader_tr_args = dataset_args['loader_tr_args']
    args.loader_te_args = dataset_args['loader_te_args']
    args.normalize = dataset_args['normalize']
    args.log = log 

    # load dataset
    X_tr, Y_tr, X_te, Y_te = get_dataset(args.dataset, args.data_path)
    if type(X_tr) is list:
        X_tr = np.array(X_tr)
        Y_tr = torch.tensor(np.array(Y_tr))
        X_te = np.array(X_te)
        Y_te = torch.tensor(np.array(Y_te))

    if type(X_tr[0]) is not np.ndarray:
        X_tr = X_tr.numpy()
        X_te = X_te.numpy()
        
    args.dim = np.shape(X_tr)[1:]
    handler = get_handler(args.dataset)

    n_pool = len(Y_tr)
    n_test = len(Y_te)

    # parameters
    if args.dataset == 'mnist':
        args.schedule = [20, 40]

    args.nEnd =  args.nEnd if args.nEnd != -1 else 100
    args.nQuery = args.nQuery if args.nQuery != -1 else (args.nEnd - args.nStart)

    NUM_INIT_LB = int(args.nStart*n_pool/100)
    NUM_QUERY = int(args.nQuery*n_pool/100) if args.nStart!= 100 else 0
    NUM_ROUND = 5
    if NUM_QUERY != 0:
        if (int(args.nEnd*n_pool/100) - NUM_INIT_LB)% NUM_QUERY != 0:
            NUM_ROUND += 1
    
    print_log("[init={:02d}] [query={:02d}] [end={:02d}]".format(NUM_INIT_LB, NUM_QUERY, int(args.nEnd*n_pool/100)), log)

    # load specified network
    # net = Model(args.model).get_model()
    

    #Initi a random value of labeled data

    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

    if args.model.lower()=="resnet50" or args.model.lower()=="resnet18":
        net=models.__dict__[args.model](n_class=args_pool[args.dataset]['n_class'])
        net.feature_extractor.conv1 = torch.nn.Conv2d(args_pool[args.dataset]['channels'], 16,kernel_size=3,stride=1,padding=1,bias=False)  
        net.discriminator.dis_fc2 = torch.nn.Linear(in_features=50, out_features=args_pool[args.dataset]['n_class'],bias=True)  
    elif args.model.lower()=="mobilenet" :
        net=models.__dict__[args.model]()
        net.conv1=torch.nn.Conv2d(args_pool[args.dataset]['channels'],32,kernel_size=3,stride=1,padding=1,bias=False)
        net.linear=torch.nn.Linear(in_features=1024, out_features=args_pool[args.dataset]['n_class'], bias=True) 
    elif args.model.lower()=="vgg" :
        net=models.__dict__[args.model]('VGG16')
        net.features[0]=torch.nn.Conv2d(args_pool[args.dataset]['channels'],64,kernel_size=3,stride=1,padding=1,bias=False)
        net.classifier=torch.nn.Linear(in_features=512, out_features=args_pool[args.dataset]['n_class'], bias=True)





   
    frameworks = framework.__dict__[args.framework](X_tr, Y_tr, X_te, Y_te, idxs_lb, net, handler, args)

    print_log('framework {} successfully loaded...'.format(args.framework), log)

    alpha = 2e-3
    # # load pretrained model
    # if args.load_ckpt:
    #     framework.load_model()
    #     idxs_lb = framework.idxs_lb
    # else:
    #     framework.train(alpha=alpha, n_epoch=args.n_epoch)
    # test_acc= framework.predict(X_te, Y_te)

    # acc = np.zeros(NUM_ROUND+1)
    # acc[0] = test_acc
    # print_log('==>> Testing accuracy {}'.format(acc[0]), log)

    strategy_al = al_methods.__dict__[args.ALstrat]
    strategy_ssl=ssl_methods.__dict__[args.SSLstrat]
    acc=frameworks.train_framework(strategy_al,strategy_ssl,NUM_ROUND,NUM_QUERY,alpha,n_epochs=args.n_epoch)

    folder_result_acc='results'
    # out_file = os.path.join(args.save_path, args.save_file)
    if not os.path.exists(folder_result_acc):
        os.mkdir(folder_result_acc)
        print(f"Folder '{folder_result_acc}' created succesfuly.")
    file_path=os.path.join(folder_result_acc,args.framework+"("+args.ALstrat+" + "+args.SSLstrat+")"+"("+args.model+" + "+args.dataset+"begin with : "+str(NUM_INIT_LB)+" Number of round"+str(NUM_ROUND)+" How many to query"+str(NUM_QUERY))
    np.save(file_path,acc)
    



if __name__ == '__main__':
    main()
