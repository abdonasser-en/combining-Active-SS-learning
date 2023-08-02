import torch
import numpy as np
import argparse
from preprocess.load_split_data import unpickle 
from models import AlexNet
import glob
from dataset import dataset

parser = argparse.ArgumentParser()

# Argument for loading the dataset
parser.add_argument("load_data", help="Dataset name or 'list' to display all available datasets", type=str)

args = parser.parse_args()

# file=[]
# dir= "datasets/cifar-10-batches-py"
# for filename in glob.iglob(f'{dir}/[dt]*'):
#     file.append(filename)

if args.load_data:
    print(dataset[0])

