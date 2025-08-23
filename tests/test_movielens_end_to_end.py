"""
End-to-End Test for TorchRS using MovieLens-like Data

This test demonstrates a complete workflow similar to TensorFlow Recommenders' 
end-to-end examples with the MovieLens dataset.
"""

import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data


def create_movielens_like_data():
    """Create MovieLens-like data for end-to-end testing."""
    # Simulate a small subset of MovieLens data
    # In a real scenario, you would load actual MovieLens data
    num_users = 100
    num_movies = 50
    num_ratings = 1000
    
    # Generate user-movie interactions
    user_ids = np.random.randint(0, num_users, num_ratings)
    movie_ids = np.random.randint(0, num_movies, num_ratings)
    ratings = np.random.randint(1, 6, num_ratings)  # Ratings from 1-5
    
    return user_ids, movie_ids, ratings


def test_movielens_end_to_end():
    """End-to-end test similar to TensorFlow Recommenders examples."""
    print("Running end-to-end MovieLens test...")
    
    # 1. Data Loading
    print("1. Loading MovieLens-like data...")
    user_ids, movie_ids, ratings = create_movielens_like_data()
    
    # Create dataset
    dataset = data.RecommendationDataset(user_ids, movie_ids, ratings)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Model Creation
    print("2. Creating user and movie models...")
    # User model - embedding followed by projection
    user_model = nn.Sequential(
        nn.Embedding(100, 64),  # User embedding
        nn.Linear(64, 32),      # Projection to 32 dimensions
        nn.ReLU()
    )
    
    # Movie model - embedding followed by projection
    movie_model = nn.Sequential(
        nn.Embedding(50, 64),   # Movie embedding
        nn.Linear(64, 32),      # Projection to 32 dimensions
        nn.ReLU()
    )
    
    # 3. Task Definition
    print("3. Defining retrieval task...")
    retrieval_task = tasks.Retrieval(
        metrics=[metrics.FactorizedTopK(k=10)]
    )
    
    # 4. Model Assembly
    print("4. Assembling full model...")
    model = models.RetrievalModel(
        user_model=user_model,
        item_model=movie_model,
        task=retrieval_task
    )
    
    # 5. Training
    print("5. Training model...")
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            user_ids = batch['user_id']
            movie_ids = batch['item_id']
            
            # Compute loss
            loss = model.compute_loss(user_ids, movie_ids, movie_ids)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    # 6. Evaluation
    print("6. Evaluating model...")
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch['user_id']
            movie_ids = batch['item_id']
            ratings = batch['rating']
            
            # Compute test loss
            test_loss = model.compute_loss(user_ids, movie_ids, movie_ids)
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"   Average Test Loss: {avg_test_loss:.4f}")
    
    # 7. Recommendation Generation
    print("7. Generating recommendations...")
    with torch.no_grad():
        # Generate recommendations for a sample user
        sample_user = torch.tensor([42])  # User ID 42
        candidate_movies = torch.arange(0, 20)  # First 20 movies as candidates
        
        top_movies, top_scores = model.recommend(sample_user, candidate_movies, k=5)
        print(f"   Top 5 recommendations for user 42:")
        for i, (movie, score) in enumerate(zip(top_movies[0], top_scores[0])):
            print(f"     {i+1}. Movie {movie.item()} (Score: {score.item():.4f})")
    
    print("End-to-end test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_movielens_end_to_end()
    if not success:
        exit(1)