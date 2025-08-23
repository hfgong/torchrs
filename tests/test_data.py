import torch
import numpy as np
from torchrs.data import RecommendationDataset, negative_sampling


def test_recommendation_dataset():
    """Test RecommendationDataset functionality."""
    # Create sample data
    user_ids = [1, 2, 3, 4, 5]
    item_ids = [10, 20, 30, 40, 50]
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Create dataset without ratings
    dataset_no_ratings = RecommendationDataset(user_ids, item_ids)
    assert len(dataset_no_ratings) == 5
    sample = dataset_no_ratings[0]
    assert 'user_id' in sample
    assert 'item_id' in sample
    assert 'rating' not in sample
    
    # Create dataset with ratings
    dataset_with_ratings = RecommendationDataset(user_ids, item_ids, ratings)
    assert len(dataset_with_ratings) == 5
    sample = dataset_with_ratings[0]
    assert 'user_id' in sample
    assert 'item_id' in sample
    assert 'rating' in sample


def test_negative_sampling():
    """Test negative sampling functionality."""
    # Create sample data
    user_ids = [1, 2, 3, 4, 5]
    item_ids = [10, 20, 30, 40, 50]
    
    # Create dataset
    dataset = RecommendationDataset(user_ids, item_ids)
    
    # Apply negative sampling
    dataset_with_negatives = negative_sampling(dataset, num_negatives=2)
    
    # Check that dataset size has increased
    assert len(dataset_with_negatives) > len(dataset)
    # Original 5 samples + 10 negative samples = 15 total
    assert len(dataset_with_negatives) == 15


if __name__ == "__main__":
    test_recommendation_dataset()
    test_negative_sampling()
    print("All data tests passed!")