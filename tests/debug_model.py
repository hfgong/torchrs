"""
Debug script to understand why recommendation scores are zero.
"""

import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data


def debug_model():
    """Debug the model to understand why scores are zero."""
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create a simple model
    user_model = nn.Sequential(
        nn.Embedding(10, 32),
        nn.Linear(32, 16),
        nn.ReLU()
    )
    
    movie_model = nn.Sequential(
        nn.Embedding(20, 32),
        nn.Linear(32, 16),
        nn.ReLU()
    )
    
    # Create a sample user and movies
    user_input = torch.tensor([0])  # User 0
    movie_inputs = torch.arange(0, 5)  # First 5 movies
    
    print("User input:", user_input)
    print("Movie inputs:", movie_inputs)
    
    # Get embeddings
    user_emb = user_model(user_input)
    movie_embs = movie_model(movie_inputs)
    
    print("User embedding shape:", user_emb.shape)
    print("Movie embeddings shape:", movie_embs.shape)
    print("User embedding sample:", user_emb[0, :5])  # First 5 values
    print("Movie embedding sample:", movie_embs[0, :5])  # First 5 values of first movie
    
    # Compute scores
    scores = torch.matmul(user_emb, movie_embs.t())
    print("Scores:", scores)
    print("Scores shape:", scores.shape)
    
    # Check if any embeddings are zero
    print("User embedding norm:", torch.norm(user_emb))
    print("Movie embeddings norms:", torch.norm(movie_embs, dim=1))


if __name__ == "__main__":
    debug_model()