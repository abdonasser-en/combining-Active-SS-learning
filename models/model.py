import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet50, vit_b_32, efficientnet_b7, ResNet50_Weights, ViT_B_32_Weights, EfficientNet_B7_Weights

class Model:

    def __init__(self, model_name, weights=None) -> None:
        """
        Initialize the Model object.

        Args:
        - model_name (str): The name of the model to be used.
        - weights (str or None): The weights to be used for the model (default: None).
        """
        self.list_models = ['resnet50', 'vit_b_32', 'efficientnet_b7']
        self.list_weights = [None, 'ResNet50_Weights', 'ViT_B_32_Weights', 'EfficientNet_B7_Weights']
        
        if model_name not in self.list_models:
            raise ValueError(f'Sorry, only these models {self.list_models} are available.')
        
        if weights not in self.list_weights:
            raise ValueError(f'Sorry, only these weights {self.list_weights} are available for models.')

        self.model_name = model_name
        self.weights = weights

    def __str__(self) -> str:
        """
        Return a string representation of the Model object.
        """
        if self.weights:
            return f'{self.model_name} is created without pre-trained weights'
        else:
            return f'{self.model_name} is created with pre-trained weights'
        
    def get_model(self):
        """
        Get the specified model based on the model_name and weights.

        Returns:
            - model (object): The model object with specified weights (if any).
        """
        if not self.weights:
            
            if self.model_name == 'resnet50':
                model = resnet50(weights=None)
            elif self.model_name == 'vit_b_32':
                model = vit_b_32(weights=None)
            elif self.model_name == 'efficientnet_b7':
                model = efficientnet_b7(weights=None)
        else:
            
            if self.model_name == 'resnet50':
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
            elif self.model_name == 'vit_b_32':
                model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            elif self.model_name == 'efficientnet_b7':
                model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)

        return model
    
