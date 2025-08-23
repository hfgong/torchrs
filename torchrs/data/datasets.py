import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RecommendationDataset(Dataset):
    """Base dataset class for recommendation systems.
    
    This class provides a foundation for recommendation datasets
    with support for user-item interactions. It can handle both
    explicit feedback (ratings) and implicit feedback (interactions).
    
    The dataset stores user IDs, item IDs, and optional ratings or
    other target variables. It's designed to work seamlessly with
    PyTorch's DataLoader for efficient batching and shuffling.
    
    Common use cases:
    - Collaborative filtering with explicit ratings
    - Implicit feedback models (clicks, views, purchases)
    - Content-based recommendation with user/item features
    """
    
    def __init__(self, user_ids, item_ids, ratings=None):
        """Initialize the recommendation dataset.
        
        Args:
            user_ids (array-like): List or array of user IDs. These can be
                integers, strings, or any hashable type that uniquely identifies users.
            item_ids (array-like): List or array of item IDs. These can be
                integers, strings, or any hashable type that uniquely identifies items.
            ratings (array-like, optional): List or array of ratings or interaction
                strengths. If provided, this enables explicit feedback models.
                If None, the dataset is suitable for implicit feedback models
                where the presence of an interaction is the target.
        """
        # Convert to PyTorch tensors for efficient processing
        # Using long tensors for IDs to support embedding layers
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        
        # Ratings are optional - if not provided, we have implicit feedback data
        if ratings is not None:
            # For explicit feedback, convert ratings to float tensors
            self.ratings = torch.tensor(ratings, dtype=torch.float)
        else:
            self.ratings = None
        
    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns:
            int: The total number of user-item interactions in the dataset.
        """
        return len(self.user_ids)
        
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Retrieves a single user-item interaction (and optionally rating)
        from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing the sample data:
                - 'user_id': User ID for this interaction
                - 'item_id': Item ID for this interaction
                - 'rating': Rating/strength (only if ratings were provided)
        """
        # Create base sample with user and item IDs
        sample = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx]
        }
        
        # Add rating if available
        if self.ratings is not None:
            sample['rating'] = self.ratings[idx]
            
        return sample


def negative_sampling(dataset, num_negatives=1):
    """Generate negative samples for a dataset.
    
    Creates negative samples by pairing users with items they haven't
    interacted with. This is essential for training retrieval models
    that need to distinguish between positive and negative examples.
    
    Note: This is a simplified implementation that randomly samples items.
    In practice, you'd want to ensure negatives aren't actually positive
    and may want to use more sophisticated negative sampling strategies.
    
    Args:
        dataset (RecommendationDataset): The dataset to sample from.
            Should contain user_ids and item_ids tensors.
        num_negatives (int): Number of negative samples to generate
            for each positive sample. Higher values may improve training
            but increase computational cost.
            
    Returns:
        RecommendationDataset: New dataset with both positive and
            negative samples combined.
    """
    # Extract the raw data from the dataset
    user_ids = dataset.user_ids.tolist()
    item_ids = dataset.item_ids.tolist()
    
    # Generate negative samples
    negative_user_ids = []
    negative_item_ids = []
    
    # Get unique items for sampling
    unique_items = list(set(item_ids))
    
    # For each positive user-item pair, generate negative samples
    for user_id in user_ids:
        for _ in range(num_negatives):
            # Add the user ID (same as positive sample)
            negative_user_ids.append(user_id)
            # Randomly sample an item (likely negative, but not guaranteed)
            negative_item_id = np.random.choice(unique_items)
            negative_item_ids.append(negative_item_id)
    
    # Combine positive and negative samples
    all_user_ids = user_ids + negative_user_ids
    all_item_ids = item_ids + negative_item_ids
    
    # Return new dataset with combined samples
    return RecommendationDataset(all_user_ids, all_item_ids)