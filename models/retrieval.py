import torch
import torch.nn as nn
from .base import Model
from ..tasks import Retrieval


class RetrievalModel(Model):
    """Two-tower retrieval model for recommendation systems.
    
    This model uses separate towers for users and items to learn embeddings
    in a shared vector space. During training, the goal is to bring embeddings
    of users and items they interact with closer together while pushing apart
    embeddings of users and items they don't interact with.
    
    The model supports both retrieval tasks (finding candidate items) and
    can generate recommendations by computing similarity scores between
    user and item embeddings.
    
    Architecture:
        User Features -> User Tower -> User Embeddings
        Item Features -> Item Tower -> Item Embeddings
        Similarity(User Embeddings, Item Embeddings) -> Scores
    """
    
    def __init__(self, user_model, item_model, task=None):
        """Initialize the retrieval model.
        
        Args:
            user_model (nn.Module): Model for encoding user features into embeddings.
                This is typically a UserTower or similar module that processes
                user features (IDs, demographics, behavior) into dense vectors.
            item_model (nn.Module): Model for encoding item features into embeddings.
                This is typically an ItemTower or similar module that processes
                item features (IDs, categories, descriptions) into dense vectors.
            task (Retrieval, optional): Task to use for training. If not provided,
                a default Retrieval task will be created. The task defines the
                loss function and metrics for training.
        """
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = task or Retrieval()
        
    def forward(self, user_inputs, item_inputs):
        """Forward pass through the retrieval model.
        
        Processes user and item inputs through their respective towers to
        produce embeddings. These embeddings can then be used to compute
        similarity scores for recommendation or training.
        
        Args:
            user_inputs (torch.Tensor): Inputs for the user tower. This could
                be user IDs, feature vectors, or other user-specific data.
            item_inputs (torch.Tensor): Inputs for the item tower. This could
                be item IDs, feature vectors, or other item-specific data.
                
        Returns:
            tuple: A tuple containing:
                - user_embeddings (torch.Tensor): User representations
                - item_embeddings (torch.Tensor): Item representations
        """
        user_embeddings = self.user_model(user_inputs)
        item_embeddings = self.item_model(item_inputs)
        return user_embeddings, item_embeddings
        
    def compute_loss(self, user_inputs, item_inputs, positive_items, negative_items=None):
        """Compute the retrieval loss.
        
        Computes the loss for training the retrieval model. This typically involves
        a sampled softmax or similar loss that encourages positive user-item pairs
        to have higher similarity scores than negative pairs.
        
        Args:
            user_inputs (torch.Tensor): Inputs for the user tower (e.g., user IDs).
            item_inputs (torch.Tensor): Inputs for the item tower (e.g., item IDs).
            positive_items (torch.Tensor): Indices of positive items (items that
                users actually interacted with). Used for computing positive scores.
            negative_items (torch.Tensor, optional): Embeddings for negative items
                (items that users did not interact with). If not provided, the
                task will sample negative items internally.
                
        Returns:
            torch.Tensor: Computed loss value for backpropagation.
        """
        user_embeddings, item_embeddings = self.forward(user_inputs, item_inputs)
        return self.task.compute_loss(user_embeddings, item_embeddings, positive_items, negative_items)
        
    def call_to_action(self, user_inputs):
        """Generate recommendations for a user.
        
        Produces user embeddings that can be used for generating recommendations.
        This method is typically used during inference to get user representations
        that can be compared against item embeddings.
        
        Args:
            user_inputs (torch.Tensor): Inputs for the user tower (e.g., user IDs).
            
        Returns:
            torch.Tensor: User embeddings for recommendation generation.
        """
        return self.user_model(user_inputs)
        
    def recommend(self, user_inputs, candidate_items, k=10):
        """Generate top-K recommendations for a user.
        
        Generates recommendations by computing similarity scores between the
        user's embedding and embeddings of candidate items, then returning
        the top-K items with the highest scores.
        
        Args:
            user_inputs (torch.Tensor): Inputs for the user tower (e.g., user ID).
                Shape: (batch_size,) or (batch_size, user_feature_dim)
            candidate_items (torch.Tensor): Items to score and rank. This can
                be item IDs or item feature tensors.
                Shape: (num_candidates,) or (num_candidates, item_feature_dim)
            k (int): Number of recommendations to return. Defaults to 10.
                
        Returns:
            tuple: A tuple containing:
                - top_k_items (torch.Tensor): Indices/IDs of top-K recommended items.
                  Shape: (batch_size, k)
                - top_k_scores (torch.Tensor): Similarity scores for recommended items.
                  Shape: (batch_size, k)
        """
        # Get user embeddings for the given user inputs
        user_embeddings = self.call_to_action(user_inputs)
        
        # Handle case where candidate_items might be indices or actual inputs
        if candidate_items.dim() == 1:
            # candidate_items are indices, pass them through item_model
            item_embeddings = self.item_model(candidate_items)
        else:
            # candidate_items are already inputs, process them directly
            item_embeddings = self.item_model(candidate_items)
        
        # Compute similarity scores using dot product
        # user_embeddings: [batch_size, embedding_dim]
        # item_embeddings: [num_candidates, embedding_dim]
        # scores: [batch_size, num_candidates]
        scores = torch.matmul(user_embeddings, item_embeddings.t())
        
        # Get top-K items for each user based on similarity scores
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        
        # Convert indices back to item IDs if needed
        if candidate_items.dim() == 1:
            # Return the actual item IDs
            top_k_items = candidate_items[top_k_indices]
        else:
            # Return the indices
            top_k_items = top_k_indices
            
        return top_k_items, top_k_scores