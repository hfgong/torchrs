"""
Simple test to isolate the scoring issue.
"""

import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data


def simple_test():
    """Simple test to isolate the scoring issue."""
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create models exactly as in the consistency test
    user_model = nn.Sequential(
        nn.Embedding(10, 32),   # 10 users, 32-dim embeddings
        nn.Linear(32, 16),      # Project to 16 dimensions
        nn.ReLU()
    )
    
    movie_model = nn.Sequential(
        nn.Embedding(20, 32),   # 20 movies, 32-dim embeddings
        nn.Linear(32, 16),      # Project to 16 dimensions
        nn.ReLU()
    )
    
    # Test the models BEFORE any training
    print("=== Before training ===")
    user_input = torch.tensor([0])
    movie_inputs = torch.arange(0, 5)
    
    user_emb_before = user_model(user_input)
    movie_embs_before = movie_model(movie_inputs)
    
    scores_before = torch.matmul(user_emb_before, movie_embs_before.t())
    print(f"Scores before training: {scores_before}")
    
    # Now let's do a minimal training step
    print("\n=== Minimal training ===")
    optimizer = torch.optim.SGD(list(user_model.parameters()) + list(movie_model.parameters()), lr=0.01)
    
    # Create a simple loss - make user 0 prefer movie 0
    target_scores = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])  # Want user 0 to prefer movie 0
    
    for i in range(5):
        optimizer.zero_grad()
        user_emb = user_model(user_input)
        movie_embs = movie_model(movie_inputs)
        scores = torch.matmul(user_emb, movie_embs.t())
        
        loss = torch.nn.functional.mse_loss(scores, target_scores)
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}, Scores: {scores}")
    
    # Test after training
    print("\n=== After training ===")
    user_emb_after = user_model(user_input)
    movie_embs_after = movie_model(movie_inputs)
    
    scores_after = torch.matmul(user_emb_after, movie_embs_after.t())
    print(f"Scores after training: {scores_after}")
    
    # Check if embeddings changed
    print(f"User embedding changed: {not torch.allclose(user_emb_before, user_emb_after)}")
    print(f"Movie embeddings changed: {not torch.allclose(movie_embs_before, movie_embs_after)}")


if __name__ == "__main__":
    simple_test()