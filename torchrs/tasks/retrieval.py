import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Task


class Retrieval(Task):
    """Retrieval task for recommendation systems.
    
    This task is used for candidate selection in two-tower models,
    typically using sampled softmax loss or similar approaches.
    """
    
    def __init__(self, loss_fn=None, metrics=None):
        """Initialize the retrieval task.
        
        Args:
            loss_fn: Loss function to use (default: sampled softmax)
            metrics: Metrics to compute during evaluation
        """
        super().__init__()
        self.loss_fn = loss_fn or self._sampled_softmax_loss
        self.metrics = metrics or []
        
    def compute_loss(self, user_embeddings, item_embeddings, positive_items, negative_items=None):
        """Compute the retrieval loss.
        
        Args:
            user_embeddings: Embeddings for users
            item_embeddings: Embeddings for positive items
            positive_items: Indices of positive items
            negative_items: Indices of negative items (optional)
            
        Returns:
            Computed loss value
        """
        return self.loss_fn(user_embeddings, item_embeddings, positive_items, negative_items)
        
    def compute_metrics(self, user_embeddings, item_embeddings, positive_items):
        """Compute retrieval metrics.
        
        Args:
            user_embeddings: Embeddings for users
            item_embeddings: Embeddings for items
            positive_items: Indices of positive items
            
        Returns:
            Dictionary of computed metrics
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(user_embeddings, item_embeddings, positive_items)
        return results
        
    def _sampled_softmax_loss(self, user_embeddings, item_embeddings, positive_items, negative_items=None):
        """Compute sampled softmax loss.
        
        Args:
            user_embeddings: Embeddings for users
            item_embeddings: Embeddings for positive items
            positive_items: Indices of positive items
            negative_items: Indices of negative items (optional)
            
        Returns:
            Computed sampled softmax loss
        """
        # Compute positive scores
        positive_scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        if negative_items is not None:
            # Compute negative scores
            negative_scores = torch.sum(
                user_embeddings.unsqueeze(1) * negative_items, dim=2
            )
            
            # Concatenate positive and negative scores
            scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
        else:
            # If no negative items provided, use all items
            scores = torch.matmul(user_embeddings, item_embeddings.t())
            
        # Compute softmax loss
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores, labels)
        
        return loss