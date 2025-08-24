import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Task


class Retrieval(Task):
    """Retrieval task for recommendation systems.
    
    This task is used for candidate selection in two-tower models,
    typically using sampled softmax loss or similar approaches.
    
    The retrieval task trains models to distinguish between items
    that users have interacted with (positive samples) and items
    they haven't (negative samples). The goal is to learn embeddings
    where user and positive item embeddings are close together in
    the vector space, while user and negative item embeddings are
    far apart.
    
    Common use cases:
    - Candidate generation for large item catalogs
    - Initial filtering before ranking stages
    - Content-based or collaborative filtering recommendations
    """
    
    def __init__(self, loss_fn=None, metrics=None, num_negatives=4):
        """Initialize the retrieval task.
        
        Args:
            loss_fn (callable, optional): Loss function to use. If not provided,
                a default sampled softmax loss function will be used. The loss
                function should accept user embeddings, item embeddings,
                positive items, and negative items as arguments.
            metrics (list, optional): Metrics to compute during evaluation.
                These could include FactorizedTopK, Recall, or other retrieval
                metrics. Each metric should be callable and return a scalar value.
            num_negatives (int): Number of negative samples to use when not
                explicitly provided. This is used in the default loss function
                when negative items are not supplied during training.
        """
        super().__init__()
        self.loss_fn = loss_fn or self._sampled_softmax_loss
        self.metrics = metrics or []
        self.num_negatives = num_negatives
        
    def compute_loss(self, user_embeddings, item_embeddings, positive_items, negative_items=None):
        """Compute the retrieval loss.
        
        Computes the loss for training the retrieval model. This typically involves
        contrasting positive user-item pairs against negative pairs to learn
        meaningful embeddings.
        
        Args:
            user_embeddings (torch.Tensor): Embeddings for users. Shape: (batch_size, embedding_dim)
            item_embeddings (torch.Tensor): Embeddings for positive items. Shape: (batch_size, embedding_dim)
            positive_items (torch.Tensor): Indices of positive items. Shape: (batch_size,)
            negative_items (torch.Tensor, optional): Embeddings for negative items.
                Shape: (batch_size, num_negative, embedding_dim). If not provided,
                the loss function will sample negative items internally.
            
        Returns:
            torch.Tensor: Computed loss value for backpropagation.
        """
        return self.loss_fn(user_embeddings, item_embeddings, positive_items, negative_items)
        
    def compute_metrics(self, user_embeddings, item_embeddings, positive_items):
        """Compute retrieval metrics.
        
        Computes evaluation metrics for the retrieval task. These metrics
        measure how well the model ranks positive items compared to other items.
        
        Args:
            user_embeddings (torch.Tensor): Embeddings for users
            item_embeddings (torch.Tensor): Embeddings for items
            positive_items (torch.Tensor): Indices of positive items
            
        Returns:
            dict: Dictionary of computed metrics (metric_name -> value)
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(user_embeddings, item_embeddings, positive_items)
        return results
        
    def _sampled_softmax_loss(self, user_embeddings, item_embeddings, positive_items, negative_items=None):
        """Compute sampled softmax loss.
        
        Implements sampled softmax loss for retrieval tasks. This loss function
        treats retrieval as a classification problem where the model must
        identify the correct item among a set of candidates (positive item
        and sampled negative items).
        
        The loss encourages the similarity score between user and positive
        item embeddings to be higher than scores with negative items.
        
        Args:
            user_embeddings (torch.Tensor): Embeddings for users. Shape: (batch_size, embedding_dim)
            item_embeddings (torch.Tensor): Embeddings for positive items. Shape: (batch_size, embedding_dim)
            positive_items (torch.Tensor): Indices of positive items. Shape: (batch_size,)
            negative_items (torch.Tensor, optional): Embeddings for negative items.
                Shape: (batch_size, num_negative, embedding_dim). If not provided,
                in-batch negative sampling will be used.
            
        Returns:
            torch.Tensor: Computed sampled softmax loss.
        """
        # Compute positive scores (element-wise product followed by sum)
        # This computes the dot product between user and positive item embeddings
        # Shape: (batch_size,)
        positive_scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        
        if negative_items is not None:
            # Compute negative scores when explicitly provided
            # negative_items shape: (batch_size, num_negative, embedding_dim)
            # user_embeddings.unsqueeze(1) shape: (batch_size, 1, embedding_dim)
            # Result shape: (batch_size, num_negative)
            negative_scores = torch.sum(
                user_embeddings.unsqueeze(1) * negative_items, dim=2
            )
            
            # Concatenate positive and negative scores for classification
            # positive_scores.unsqueeze(1) shape: (batch_size, 1)
            # negative_scores shape: (batch_size, num_negative)
            # scores shape: (batch_size, 1 + num_negative)
            scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
        else:
            # If no negative items provided, use in-batch negative sampling
            # This is a common approach when explicit negatives are not available
            batch_size = user_embeddings.size(0)
            
            # Compute all pairwise scores between users and items in the batch
            # user_embeddings: (batch_size, embedding_dim)
            # item_embeddings: (batch_size, embedding_dim)
            # all_scores: (batch_size, batch_size) where [i,j] is score of user i with item j
            all_scores = torch.matmul(user_embeddings, item_embeddings.t())
            
            # For sampled softmax, we want the positive item at index 0
            # Extract diagonal elements (user i with their positive item i)
            positive_scores_for_loss = torch.diag(all_scores).unsqueeze(1)  # (batch_size, 1)
            
            # Use other items in the batch as negative samples
            # For simplicity, take the first few non-diagonal elements as negatives
            negative_scores_simple = []
            for i in range(min(self.num_negatives, batch_size-1)):
                # Take the score of user i with item (i+1) % batch_size as a negative sample
                neg_idx = (torch.arange(batch_size) + i + 1) % batch_size
                neg_scores = all_scores[torch.arange(batch_size), neg_idx].unsqueeze(1)
                negative_scores_simple.append(neg_scores)
            
            if negative_scores_simple:
                # Combine positive score with negative scores
                negative_scores_combined = torch.cat(negative_scores_simple, dim=1)
                scores = torch.cat([positive_scores_for_loss, negative_scores_combined], dim=1)
            else:
                # Fallback if we can't create negative samples
                scores = positive_scores_for_loss
                
        # Compute softmax loss with cross-entropy
        # The positive item should be at index 0 in the scores tensor
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores, labels)
        
        return loss