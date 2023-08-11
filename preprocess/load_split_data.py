import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import Subset, DataLoader,random_split



class LoadDataset:
    def __init__(self, dataset_name, root="./datasets", transform=ToTensor(), download=True, shuffle=True,ratio_uldata=0.01,val_size=0.1):
        """
        Initialize the DataLoader.

        Args:
            - dataset_name (str): Name of the dataset.
            - root (str, optional): Root directory for the dataset. Defaults to "./datasets".
            - transform (object, optional): Transformation to apply to the dataset. Defaults to ToTensor().
            - download (bool, optional): Whether to download the dataset. Defaults to True.
            - shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        """
        self.root = root
        self.dataset_name = dataset_name
        self.transform = transform
        self.download = download
        self.shuffle = shuffle
        self.ratio_uldata=ratio_uldata
        self.val_size=val_size

    def load_dataset(self):
        """
        Load the specified dataset.

        Returns:
            - tuple or list: The loaded dataset.
        """
        class_names = [
            "MNIST", "CIFAR10", "EuroSAT", "ImageNet", 
        ]

        if self.dataset_name in class_names:
            data_att = getattr(datasets, self.dataset_name, None)
            if self.dataset_name in ['EuroSAT']:
                train_data = data_att(root=self.root, download=self.download, transform=self.transform)
                return train_data
            
            elif self.dataset_name in [ 'ImageNet']:
                    train_data = data_att(root=self.root, split='train', download=self.download, transform=self.transform)
                    test_data = data_att(root=self.root, split='val', download=self.download, transform=self.transform)
                    return (train_data, test_data)  
            else:
                train_data = data_att(root=self.root, train=True, download=self.download, transform=self.transform)
                test_data = data_att(root=self.root, train=False, download=self.download, transform=self.transform)
                return (train_data, test_data)
            
        elif self.dataset_name == "list":
            return class_names
        
        else:
            raise ValueError(f'{self.dataset_name} is an invalid dataset name. For supported datasets, please run: python dataset.py list')

    def split_dataset(self, dataset,val_size = 0.1, test_size=0.1, random_state=42):
        """
        Split a dataset into training and test sets.

        Args:
            - dataset (torchvision.datasets.Dataset): The dataset to be split.
            - test_size (float, optional): The proportion of the dataset to include in the test set.
                Defaults to 0.2.
            - random_state (int or RandomState, optional): Seed used by the random number generator.
                Defaults to 42.

        Returns:
            - tuple: A tuple containing the training dataset and test dataset.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size should be a float between 0 and 1.")

        if self.dataset_name == 'EuroSAT':
            total_samples = len(dataset)

            indices = list(range(total_samples))
            train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            return (train_dataset.dataset, test_dataset.dataset)
            

        else:
            print(f'The dataset {self.dataset_name} is already split, no further split needed.')
            return dataset
        

        
    def split_val(self,train_dataset,val_size = 0.1):

        train_size = int((1-self.val_size) * len(train_dataset))  # 80% pour l'entraînement
        val_size = len(train_dataset) - train_size  # 20% pour la validation
        train_data, val_data = random_split(train_dataset, [train_size, val_size])
        return train_data, val_data


    
    def idx_labeled(self, data):

        init_sample = int(0.01 * len(data))
        train_indices = torch.arange(len(data))
        index = torch.randperm(len(data))[:init_sample]

        idx_lb = torch.isin(train_indices, index)

        return idx_lb

    def sep_data_label(self,data):
        batch_size = len(data)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        X,y = next(iter(data_loader ))
    
        return X,y
    
    
    def conv_tensor_numpy(self,data):
        return data.numpy()




    def ratio_unlabeled_data(self,train_data):
        """
        Split the given 'train_data' into labeled and unlabeled subsets based on the provided ratio.

        Args:
            - train_data (torch.utils.data.Dataset): The entire dataset to be split into labeled and unlabeled subsets.
            - ratio_uldata (float, optional): The ratio of unlabeled data to the entire dataset. Default is 0.01.

        Returns:
            - tuple: A tuple containing two subsets of 'train_data'.
                The first subset is the labeled data.
                The second subset is the unlabeled data.
                The init sample index of labeled data.
        """


        init_sample = int(self.ratio_uldata * len(train_data))

        # labeled data
        labeled_indices = torch.randperm(len(train_data))[:init_sample]
        train_labeled_data = Subset(train_data, labeled_indices)

        #unlabeled data
        pool_indices = torch.randperm(len(train_data))[init_sample:]
        train_unlabeled_data = Subset(train_data, pool_indices)

        return train_labeled_data, train_unlabeled_data, init_sample

    



