import torch
import torch.nn as nn


class Model(nn.Module):
    """Base class for recommendation models.
    
    This class extends PyTorch's nn.Module to provide a foundation
    for building recommendation systems with standardized interfaces
    for training and evaluation.
    
    The base model provides abstract methods that subclasses should
    implement for specific recommendation tasks like retrieval or ranking.
    """
    
    def __init__(self):
        """Initialize the base recommendation model.
        
        Calls the parent nn.Module constructor to properly initialize
        the PyTorch module infrastructure.
        """
        super().__init__()
        
    def compute_loss(self, inputs, targets):
        """Compute the loss for the model.
        
        This method should be overridden by subclasses to implement
        task-specific loss computation. For example, retrieval models
        might use sampled softmax loss, while ranking models might
        use mean squared error.
        
        Args:
            inputs: Model inputs (e.g., user IDs, item IDs)
            targets: Ground truth targets (e.g., ratings, labels)
            
        Returns:
            Computed loss value as a tensor
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("compute_loss method must be implemented by subclasses")
        
    def call_to_action(self, inputs):
        """Perform inference on the model.
        
        This method should be overridden by subclasses to implement
        task-specific inference logic. For example, generating
        recommendations for a user or predicting item scores.
        
        Args:
            inputs: Model inputs for inference
            
        Returns:
            Model predictions or recommendations
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("call_to_action method must be implemented by subclasses")