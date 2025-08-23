import torch
import pytest
from torchrs.models import Embedding, RetrievalModel
from torchrs.tasks import Retrieval
from torchrs.metrics import FactorizedTopK


def test_embedding_creation():
    """Test creation of embedding layers."""
    embedding = Embedding(num_embeddings=100, embedding_dim=32)
    assert embedding.num_embeddings == 100
    assert embedding.embedding_dim == 32
    
    # Test forward pass
    indices = torch.tensor([1, 5, 10])
    output = embedding(indices)
    assert output.shape == (3, 32)


def test_retrieval_model():
    """Test creation and basic functionality of retrieval model."""
    # Create simple user and item models
    user_model = torch.nn.Embedding(100, 32)
    item_model = torch.nn.Embedding(50, 32)
    
    # Create retrieval model
    model = RetrievalModel(user_model, item_model)
    
    # Test forward pass
    user_ids = torch.tensor([1, 2, 3])
    item_ids = torch.tensor([4, 5, 6])
    
    user_embeddings, item_embeddings = model(user_ids, item_ids)
    assert user_embeddings.shape == (3, 32)
    assert item_embeddings.shape == (3, 32)


def test_factorized_topk():
    """Test FactorizedTopK metric."""
    # Create sample embeddings
    user_embeddings = torch.randn(10, 32)
    item_embeddings = torch.randn(20, 32)
    positive_items = torch.arange(10)  # Each user's positive item is at index equal to user ID
    
    # Create metric
    metric = FactorizedTopK(k=5)
    
    # Compute metric
    result = metric(user_embeddings, item_embeddings, positive_items)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


if __name__ == "__main__":
    test_embedding_creation()
    test_retrieval_model()
    test_factorized_topk()
    print("All tests passed!")