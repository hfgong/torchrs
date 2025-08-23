import torch


class FactorizedTopK:
    """Factorized Top-K metric for retrieval evaluation.
    
    This metric evaluates retrieval performance by checking how often
    the true item appears in the top-K recommendations.
    """
    
    def __init__(self, k=10, candidates=None):
        """Initialize the FactorizedTopK metric.
        
        Args:
            k: Number of top recommendations to consider
            candidates: Candidate items for retrieval (optional)
        """
        self.k = k
        self.candidates = candidates
        self.name = f"factorized_top_{k}"
        
    def __call__(self, user_embeddings, item_embeddings, positive_items):
        """Compute the FactorizedTopK metric.
        
        Args:
            user_embeddings: Embeddings for users
            item_embeddings: Embeddings for items
            positive_items: Indices of positive items
            
        Returns:
            Computed metric value
        """
        # Compute scores between users and items
        scores = torch.matmul(user_embeddings, item_embeddings.t())
        
        # Get top-K items for each user
        _, top_k_indices = torch.topk(scores, self.k, dim=1)
        
        # Check if positive items are in top-K
        positive_items_expanded = positive_items.unsqueeze(1)
        correct_predictions = (top_k_indices == positive_items_expanded).any(dim=1)
        
        # Compute accuracy
        accuracy = correct_predictions.float().mean()
        
        return accuracy.item()