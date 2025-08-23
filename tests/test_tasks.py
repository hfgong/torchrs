import torch
import pytest
from torchrs.tasks import Retrieval, Ranking


def test_retrieval_task():
    """Test retrieval task functionality."""
    task = Retrieval()
    
    # Create sample embeddings
    user_embeddings = torch.randn(10, 32)
    item_embeddings = torch.randn(10, 32)
    positive_items = torch.arange(10)
    
    # Test loss computation
    loss = task.compute_loss(user_embeddings, item_embeddings, positive_items)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_ranking_task():
    """Test ranking task functionality."""
    task = Ranking()
    
    # Create sample predictions and targets
    predictions = torch.randn(10)
    targets = torch.randn(10)
    
    # Test loss computation
    loss = task.compute_loss(predictions, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


if __name__ == "__main__":
    test_retrieval_task()
    test_ranking_task()
    print("All task tests passed!")