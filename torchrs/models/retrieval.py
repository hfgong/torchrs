import torch
import torch.nn as nn
from .base import Model
from ..tasks import Retrieval


class RetrievalModel(Model):
    """Two-tower retrieval model for recommendation systems.
    
    This model uses separate towers for users and items to learn
    embeddings, then computes scores using dot product or other
    similarity measures.
    """
    
    def __init__(self, user_model, item_model, task=None):
        """Initialize the retrieval model.
        
        Args:
            user_model: Model for encoding user features
            item_model: Model for encoding item features
            task: Task to use for training (default: Retrieval)
        """
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = task or Retrieval()
        
    def forward(self, user_inputs, item_inputs):
        """Forward pass through the retrieval model.
        
        Args:
            user_inputs: Inputs for the user tower
            item_inputs: Inputs for the item tower
            
        Returns:
            Tuple of user embeddings and item embeddings
        """
        user_embeddings = self.user_model(user_inputs)
        item_embeddings = self.item_model(item_inputs)
        return user_embeddings, item_embeddings
        
    def compute_loss(self, user_inputs, item_inputs, positive_items, negative_items=None):
        """Compute the retrieval loss.
        
        Args:
            user_inputs: Inputs for the user tower
            item_inputs: Inputs for the item tower
            positive_items: Indices of positive items
            negative_items: Indices of negative items (optional)
            
        Returns:
            Computed loss value
        """
        user_embeddings, item_embeddings = self.forward(user_inputs, item_inputs)
        return self.task.compute_loss(user_embeddings, item_embeddings, positive_items, negative_items)
        
    def call_to_action(self, user_inputs):
        """Generate recommendations for a user.
        
        Args:
            user_inputs: Inputs for the user tower
            
        Returns:
            User embeddings for recommendation generation
        """
        return self.user_model(user_inputs)
        
    def recommend(self, user_inputs, candidate_items, k=10):
        """Generate top-K recommendations for a user.
        
        Args:
            user_inputs: Inputs for the user tower
            candidate_items: Items to score
            k: Number of recommendations to return
            
        Returns:
            Top-K recommended items and their scores
        """
        user_embeddings = self.call_to_action(user_inputs)
        item_embeddings = self.item_model(candidate_items)
        
        # Compute scores
        scores = torch.matmul(user_embeddings, item_embeddings.t())
        
        # Get top-K items
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        
        return top_k_indices, top_k_scores