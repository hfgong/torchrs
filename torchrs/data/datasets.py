import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RecommendationDataset(Dataset):
    """Base dataset class for recommendation systems.
    
    This class provides a foundation for recommendation datasets
    with support for user-item interactions.
    """
    
    def __init__(self, user_ids, item_ids, ratings=None):
        """Initialize the recommendation dataset.
        
        Args:
            user_ids: List or array of user IDs
            item_ids: List or array of item IDs
            ratings: List or array of ratings (optional)
        """
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float) if ratings is not None else None
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.user_ids)
        
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing user_id, item_id, and rating (if available)
        """
        sample = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx]
        }
        
        if self.ratings is not None:
            sample['rating'] = self.ratings[idx]
            
        return sample


def negative_sampling(dataset, num_negatives=1):
    """Generate negative samples for a dataset.
    
    Args:
        dataset: RecommendationDataset to sample from
        num_negatives: Number of negative samples per positive sample
        
    Returns:
        RecommendationDataset with negative samples added
    """
    # This is a simplified implementation
    # In practice, you'd want to ensure negatives aren't actually positive
    user_ids = dataset.user_ids.tolist()
    item_ids = dataset.item_ids.tolist()
    
    # Generate negative samples
    negative_user_ids = []
    negative_item_ids = []
    
    unique_items = list(set(item_ids))
    
    for user_id in user_ids:
        for _ in range(num_negatives):
            negative_user_ids.append(user_id)
            # Randomly sample an item that's likely negative
            negative_item_id = np.random.choice(unique_items)
            negative_item_ids.append(negative_item_id)
    
    # Combine positive and negative samples
    all_user_ids = user_ids + negative_user_ids
    all_item_ids = item_ids + negative_item_ids
    
    return RecommendationDataset(all_user_ids, all_item_ids)