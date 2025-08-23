import torch


class FactorizedTopK:
    """Factorized Top-K metric for retrieval evaluation.
    
    This metric evaluates retrieval performance by checking how often
    the true item appears in the top-K recommendations. It's a key
    metric for assessing the quality of retrieval models.
    
    The metric works by:
    1. Computing similarity scores between all users and all items
    2. Finding the top-K items for each user based on these scores
    3. Checking if the true positive items are in the top-K lists
    4. Computing the accuracy (fraction of users with correct items in top-K)
    
    Higher values indicate better retrieval performance, with 1.0 being perfect.
    This metric is particularly useful for evaluating candidate generation models.
    """
    
    def __init__(self, k=10, candidates=None):
        """Initialize the FactorizedTopK metric.
        
        Args:
            k (int): Number of top recommendations to consider. Common values
                are 5, 10, or 20 depending on the application. Lower values
                make the metric more stringent.
            candidates (torch.Tensor, optional): Candidate items for retrieval.
                If provided, only these items will be considered for ranking.
                If None, all items in the item_embeddings will be considered.
        """
        self.k = k
        self.candidates = candidates
        # Name used when returning results in a dictionary
        self.name = f"factorized_top_{k}"
        
    def __call__(self, user_embeddings, item_embeddings, positive_items):
        """Compute the FactorizedTopK metric.
        
        Evaluates how well the model ranks positive items in the top-K positions.
        
        Args:
            user_embeddings (torch.Tensor): Embeddings for users.
                Shape: (num_users, embedding_dim)
            item_embeddings (torch.Tensor): Embeddings for items.
                Shape: (num_items, embedding_dim)
            positive_items (torch.Tensor): Indices of positive items for each user.
                Shape: (num_users,) where positive_items[i] is the true item for user i.
            
        Returns:
            float: Computed metric value between 0.0 and 1.0.
                1.0 indicates perfect retrieval (all positive items in top-K).
                0.0 indicates poor retrieval (no positive items in top-K).
        """
        # Compute similarity scores between all users and all items
        # Result shape: (num_users, num_items)
        # scores[i, j] represents the similarity between user i and item j
        scores = torch.matmul(user_embeddings, item_embeddings.t())
        
        # Get top-K items for each user based on similarity scores
        # top_k_indices shape: (num_users, k)
        # Contains the indices of the k highest-scoring items for each user
        _, top_k_indices = torch.topk(scores, self.k, dim=1)
        
        # Check if positive items are in top-K for each user
        # positive_items_expanded shape: (num_users, 1)
        # correct_predictions shape: (num_users,) with boolean values
        positive_items_expanded = positive_items.unsqueeze(1)
        correct_predictions = (top_k_indices == positive_items_expanded).any(dim=1)
        
        # Compute accuracy: fraction of users with correct items in top-K
        # This gives us a single scalar value between 0.0 and 1.0
        accuracy = correct_predictions.float().mean()
        
        # Return as Python float for easier handling
        return accuracy.item()