"""
Simple comparison of recommendations for MovieLens dataset users
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import torch
import torch.nn as nn
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

def load_and_compare():
    """Load MovieLens data and compare TFRS vs TorchRS recommendations."""
    print("Loading MovieLens 100K dataset...")
    
    # Load a small subset for demonstration
    ratings = tfds.load("movielens/100k-ratings", split="train[:1000]")
    
    # Convert to lists for easier handling
    user_ids = []
    movie_ids = []
    ratings_list = []
    
    for record in ratings:
        user_ids.append(record["user_id"].numpy().decode('utf-8'))
        movie_ids.append(record["movie_id"].numpy().decode('utf-8'))
        ratings_list.append(float(record["user_rating"].numpy()))
    
    print(f"Loaded {len(user_ids)} ratings")
    print(f"Unique users: {len(set(user_ids))}")
    print(f"Unique movies: {len(set(movie_ids))}")
    
    # Show some sample data
    print("\nSample ratings:")
    for i in range(5):
        print(f"  User {user_ids[i]} rated Movie {movie_ids[i]} with {ratings_list[i]} stars")
    
    # Show unique users
    unique_users = list(set(user_ids))
    print(f"\nFirst 5 unique users: {unique_users[:5]}")
    
    print("\nNote: For a full comparison of TFRS vs TorchRS recommendations,")
    print("we would need to train both models and generate recommendations")
    print("for the same users, which requires more extensive implementation.")
    
    return user_ids, movie_ids, ratings_list

if __name__ == "__main__":
    load_and_compare()