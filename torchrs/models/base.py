import torch
import torch.nn as nn


class Model(nn.Module):
    """Base class for recommendation models.
    
    This class extends PyTorch's nn.Module to provide a foundation
    for building recommendation systems with standardized interfaces
    for training and evaluation.
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_loss(self, inputs, targets):
        """Compute the loss for the model.
        
        This method should be overridden by subclasses to implement
        task-specific loss computation.
        
        Args:
            inputs: Model inputs
            targets: Ground truth targets
            
        Returns:
            Computed loss value
        """
        raise NotImplementedError("compute_loss method must be implemented by subclasses")
        
    def call_to_action(self, inputs):
        """Perform inference on the model.
        
        This method should be overridden by subclasses to implement
        task-specific inference logic.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model predictions
        """
        raise NotImplementedError("call_to_action method must be implemented by subclasses")