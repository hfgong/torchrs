import torch
import torch.nn as nn
from typing import Union, List


class Tower(nn.Module):
    """Base tower module for two-tower architectures.
    
    This module represents one tower (either user or item) in a two-tower
    recommendation model. It processes inputs through a sequence of layers
    to produce embeddings.
    """
    
    def __init__(self, layers: Union[nn.Module, List[nn.Module]]):
        """Initialize the tower.
        
        Args:
            layers: A single module or list of modules defining the tower architecture
        """
        super().__init__()
        
        if isinstance(layers, list):
            self.tower = nn.Sequential(*layers)
        else:
            self.tower = layers
            
    def forward(self, inputs):
        """Forward pass through the tower.
        
        Args:
            inputs: Input tensor to process
            
        Returns:
            Embedding representation of the inputs
        """
        return self.tower(inputs)


class UserTower(Tower):
    """User tower for two-tower recommendation models.
    
    This tower processes user features to produce user embeddings.
    """
    
    def __init__(self, layers: Union[nn.Module, List[nn.Module]]):
        """Initialize the user tower.
        
        Args:
            layers: A single module or list of modules defining the tower architecture
        """
        super().__init__(layers)


class ItemTower(Tower):
    """Item tower for two-tower recommendation models.
    
    This tower processes item features to produce item embeddings.
    """
    
    def __init__(self, layers: Union[nn.Module, List[nn.Module]]):
        """Initialize the item tower.
        
        Args:
            layers: A single module or list of modules defining the tower architecture
        """
        super().__init__(layers)