import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Task


class Ranking(Task):
    """Ranking task for recommendation systems.
    
    This task is used for scoring and ranking items, typically using
    regression losses for rating prediction or classification losses
    for click prediction.
    """
    
    def __init__(self, loss_fn=None, metrics=None):
        """Initialize the ranking task.
        
        Args:
            loss_fn: Loss function to use (default: MSE)
            metrics: Metrics to compute during evaluation
        """
        super().__init__()
        self.loss_fn = loss_fn or nn.MSELoss()
        self.metrics = metrics or []
        
    def compute_loss(self, predictions, targets):
        """Compute the ranking loss.
        
        Args:
            predictions: Predicted scores/ratings
            targets: Ground truth scores/ratings
            
        Returns:
            Computed loss value
        """
        return self.loss_fn(predictions, targets)
        
    def compute_metrics(self, predictions, targets):
        """Compute ranking metrics.
        
        Args:
            predictions: Predicted scores/ratings
            targets: Ground truth scores/ratings
            
        Returns:
            Dictionary of computed metrics
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(predictions, targets)
        return results