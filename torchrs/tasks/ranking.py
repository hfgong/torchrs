import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Task


class Ranking(Task):
    """Ranking task for recommendation systems.
    
    This task is used for scoring and ranking items, typically using
    regression losses for rating prediction or classification losses
    for click prediction.
    
    Unlike retrieval tasks that focus on candidate selection, ranking
    tasks focus on precise scoring of items that are already candidates.
    This makes them suitable for fine-grained personalization and
    precise recommendation ordering.
    
    Common use cases:
    - Predicting explicit ratings (1-5 stars)
    - Predicting implicit feedback (click/no-click)
    - Learning-to-rank with pointwise, pairwise, or listwise losses
    - Fine-tuning recommendations from retrieval stage
    """
    
    def __init__(self, loss_fn=None, metrics=None):
        """Initialize the ranking task.
        
        Args:
            loss_fn (callable, optional): Loss function to use for training.
                If not provided, Mean Squared Error (MSE) loss will be used
                as a default. For rating prediction, MSE is appropriate.
                For click prediction, Binary Cross-Entropy might be better.
            metrics (list, optional): Metrics to compute during evaluation.
                These could include RMSE, MAE, Precision, Recall, or other
                ranking-specific metrics. Each metric should be callable
                and return a scalar value.
        """
        super().__init__()
        # Default to MSE loss for regression tasks (rating prediction)
        # For classification tasks (click prediction), BCE might be more appropriate
        self.loss_fn = loss_fn or nn.MSELoss()
        self.metrics = metrics or []
        
    def compute_loss(self, predictions, targets):
        """Compute the ranking loss.
        
        Computes the loss for training the ranking model. This measures
        how well the predicted scores match the actual target values.
        
        For rating prediction, targets are typically numerical ratings.
        For click prediction, targets are binary indicators (0 or 1).
        
        Args:
            predictions (torch.Tensor): Predicted scores/ratings.
                Shape: (batch_size,) or (batch_size, 1)
            targets (torch.Tensor): Ground truth scores/ratings.
                Shape should match predictions.
            
        Returns:
            torch.Tensor: Computed loss value for backpropagation.
        """
        return self.loss_fn(predictions, targets)
        
    def compute_metrics(self, predictions, targets):
        """Compute ranking metrics.
        
        Computes evaluation metrics for the ranking task. These metrics
        measure the quality of the predicted scores compared to targets.
        
        For regression tasks, common metrics include RMSE, MAE.
        For classification tasks, common metrics include Precision, Recall, AUC.
        
        Args:
            predictions (torch.Tensor): Predicted scores/ratings
            targets (torch.Tensor): Ground truth scores/ratings
            
        Returns:
            dict: Dictionary of computed metrics (metric_name -> value)
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(predictions, targets)
        return results