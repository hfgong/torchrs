"""
Detailed debug script to understand why scores are zero.
"""

import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data


def detailed_debug():
    """Detailed debugging of the model and scoring."""
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
    
    # Test with a single user and movie
    print("=== Testing single user/movie ===")
    user_input = torch.tensor([0])  # User 0
    movie_input = torch.tensor([0])  # Movie 0
    
    print(f"User input: {user_input}")
    print(f"Movie input: {movie_input}")
    
    # Get embeddings
    user_emb = user_model(user_input)
    movie_emb = movie_model(movie_input)
    
    print(f"User embedding shape: {user_emb.shape}")
    print(f"Movie embedding shape: {movie_emb.shape}")
    print(f"User embedding (first 5): {user_emb[0, :5]}")
    print(f"Movie embedding (first 5): {movie_emb[0, :5]}")
    
    # Compute dot product
    score = torch.dot(user_emb[0], movie_emb[0])
    print(f"Dot product score: {score}")
    
    # Test with multiple movies
    print("\n=== Testing multiple movies ===")
    movie_inputs = torch.arange(0, 5)  # Movies 0-4
    print(f"Movie inputs: {movie_inputs}")
    
    movie_embs = movie_model(movie_inputs)
    print(f"Movie embeddings shape: {movie_embs.shape}")
    
    # Compute scores using matrix multiplication
    scores = torch.matmul(user_emb, movie_embs.t())
    print(f"Scores shape: {scores.shape}")
    print(f"Scores: {scores}")
    
    # Test ReLU activation
    print("\n=== Checking ReLU activation ===")
    # Get the output before ReLU
    user_emb_no_relu = nn.Sequential(
        nn.Embedding(10, 32),
        nn.Linear(32, 16)
    )
    user_emb_no_relu[0].weight.data = user_model[0].weight.data
    user_emb_no_relu[1].weight.data = user_model[1].weight.data
    user_emb_no_relu[1].bias.data = user_model[1].bias.data
    
    user_emb_before_relu = user_emb_no_relu(user_input)
    print(f"Before ReLU (first 5): {user_emb_before_relu[0, :5]}")
    print(f"After ReLU (first 5): {user_emb[0, :5]}")
    
    # Check if all values are negative (which would make ReLU zero)
    negative_count = (user_emb_before_relu < 0).sum().item()
    total_count = user_emb_before_relu.numel()
    print(f"Negative values before ReLU: {negative_count}/{total_count}")


if __name__ == "__main__":
    detailed_debug()