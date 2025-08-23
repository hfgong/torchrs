import torch
import torch.nn as nn
from typing import Union, List


class Tower(nn.Module):
    """Base tower module for two-tower architectures.
    
    This module represents one tower (either user or item) in a two-tower
    recommendation model. It processes inputs through a sequence of layers
    to produce dense vector embeddings that capture the essential features
    of users or items.
    
    The tower architecture can be simple (single layer) or complex (multiple
    layers with nonlinearities) depending on the complexity of the features
    and relationships that need to be captured.
    
    Towers are a key component in modern recommendation systems, particularly
    for retrieval tasks where users and items are embedded into a shared
    vector space for efficient similarity computation.
    """
    
    def __init__(self, layers: Union[nn.Module, List[nn.Module]]):
        """Initialize the tower.
        
        Args:
            layers: A single PyTorch module or list of modules defining the
                tower architecture. If a list is provided, the modules will
                be wrapped in a Sequential container.
                
                Examples:
                    - Single layer: nn.Linear(100, 64)
                    - Multiple layers: [nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 32)]
        """
        super().__init__()
        
        # Handle both single module and list of modules
        if isinstance(layers, list):
            # Wrap list of layers in Sequential container for easy forward pass
            self.tower = nn.Sequential(*layers)
        else:
            # Use the provided module directly
            self.tower = layers
            
    def forward(self, inputs):
        """Forward pass through the tower.
        
        Processes the input features through the tower's layers to produce
        a dense embedding representation. The output embedding should be
        in a shared vector space with the other tower for similarity computation.
        
        Args:
            inputs (torch.Tensor): Input tensor to process. Shape depends on
                the first layer of the tower (e.g., (batch_size, input_features)).
                
        Returns:
            torch.Tensor: Embedding representation of the inputs.
                Shape depends on the last layer of the tower (e.g., (batch_size, embedding_dim)).
        """
        return self.tower(inputs)


class UserTower(Tower):
    """User tower for two-tower recommendation models.
    
    This tower processes user features (such as user ID, demographic information,
    historical behavior, etc.) to produce user embeddings. These embeddings
    represent users in a dense vector space where similar users have similar
    representations.
    
    The user tower is typically paired with an item tower in retrieval models,
    where the goal is to match user embeddings with relevant item embeddings.
    """
    
    def __init__(self, layers: Union[nn.Module, List[nn.Module]]):
        """Initialize the user tower.
        
        Args:
            layers: A single module or list of modules defining the tower architecture.
                This could be as simple as an embedding layer or as complex as
                multiple dense layers with nonlinearities.
                
                Example architectures:
                    - Simple embedding: nn.Embedding(num_users, 64)
                    - Complex network: [nn.Embedding(num_users, 128), nn.Linear(128, 64), nn.ReLU()]
        """
        super().__init__(layers)


class ItemTower(Tower):
    """Item tower for two-tower recommendation models.
    
    This tower processes item features (such as item ID, category, description,
    price, etc.) to produce item embeddings. These embeddings represent items
    in a dense vector space where similar items have similar representations.
    
    The item tower is typically paired with a user tower in retrieval models,
    where the goal is to match user embeddings with relevant item embeddings.
    The similarity between user and item embeddings determines recommendation scores.
    """
    
    def __init__(self, layers: Union[nn.Module, List[nn.Module]]):
        """Initialize the item tower.
        
        Args:
            layers: A single module or list of modules defining the tower architecture.
                This could be as simple as an embedding layer or as complex as
                multiple dense layers with nonlinearities.
                
                Example architectures:
                    - Simple embedding: nn.Embedding(num_items, 64)
                    - Complex network: [nn.Embedding(num_items, 128), nn.Linear(128, 64), nn.ReLU()]
        """
        super().__init__(layers)