import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse
import preprocess.load_split_data as pr



parser = argparse.ArgumentParser()

# Argument for loading the dataset
parser.add_argument("data", help="Dataset name or 'list' to display all available datasets", type=str)
parser
args = parser.parse_args()

# args.data
data_loader = pr.LoadDataset(args.data)
dataset = data_loader.load_dataset()


if args.data == "list":
    print("Available datasets:")
    for dataset_name in dataset:
        print(dataset_name)





# class_names = ["Caltech101", "MyClass","MNIST"]
# print("ets")

# n=input("entre ton class: ")
# print(type(n))
# print(getattr(datasets,n,None))
# class_name=getattr(datasets,n,None)
# data_image= class_name(root="./dataset", download=True)


# if __name__=="__main___":
#     class_names = ["Caltech101", "MyClass","MNIST"]
#     print("ets")

#     n=input("entre ton cclass")
#     print(type(n))
#     if n in  class_names:
#         data=datasets.n(root="./dataset")import numpy as np
