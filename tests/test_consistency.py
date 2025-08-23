"""
Consistency Test for TorchRS

This test is designed to be comparable with TensorFlow Recommenders' 
MovieLens examples to check for consistency in results.
"""

import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data


def create_standardized_movielens_data():
    """Create a standardized MovieLens-like dataset for consistency testing."""
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create a small, fixed dataset that mimics MovieLens
    # User IDs: 0-9 (10 users)
    # Movie IDs: 0-19 (20 movies)
    # Ratings: 1-5
    
    user_ids = []
    movie_ids = []
    ratings = []
    
    # Create a deterministic dataset with more interactions for better batch sampling
    for user_id in range(10):
        # Each user rates 10 random movies (more interactions)
        rated_movies = np.random.choice(20, size=10, replace=False)
        for movie_id in rated_movies:
            rating = np.random.randint(1, 6)  # Rating from 1-5
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            ratings.append(float(rating))
    
    return np.array(user_ids), np.array(movie_ids), np.array(ratings)


def run_consistency_test():
    """Run a consistency test that could be compared with TFRS."""
    print("Running consistency test...")
    
    # 1. Load standardized data
    print("1. Loading standardized data...")
    user_ids, movie_ids, ratings = create_standardized_movielens_data()
    
    print(f"   Dataset size: {len(user_ids)} interactions")
    print(f"   Users: {len(np.unique(user_ids))}")
    print(f"   Movies: {len(np.unique(movie_ids))}")
    print(f"   Rating range: {int(np.min(ratings))}-{int(np.max(ratings))}")
    
    # Create dataset
    dataset = data.RecommendationDataset(user_ids, movie_ids, ratings)
    
    # Use all data for both training and testing (for consistency)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 2. Create models with fixed architecture
    print("2. Creating models...")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # User model: Embedding -> Linear -> ReLU
    user_model = nn.Sequential(
        nn.Embedding(10, 32),   # 10 users, 32-dim embeddings
        nn.Linear(32, 16),      # Project to 16 dimensions
        nn.ReLU()
    )
    
    # Movie model: Embedding -> Linear -> ReLU
    movie_model = nn.Sequential(
        nn.Embedding(20, 32),   # 20 movies, 32-dim embeddings
        nn.Linear(32, 16),      # Project to 16 dimensions
        nn.ReLU()
    )
    
    # 3. Define task with proper loss function
    print("3. Defining task...")
    task = tasks.Retrieval(
        metrics=[metrics.FactorizedTopK(k=5)],
        num_negatives=4
    )
    
    # 4. Create model
    print("4. Creating retrieval model...")
    model = models.RetrievalModel(
        user_model=user_model,
        item_model=movie_model,
        task=task
    )
    
    # 5. Train with fixed parameters
    print("5. Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Set seed again for reproducibility
    torch.manual_seed(42)
    
    train_losses = []
    model.train()
    for epoch in range(10):  # Train for 10 epochs
        epoch_loss = 0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            user_ids_batch = batch['user_id']
            movie_ids_batch = batch['item_id']
            
            # Compute loss
            loss = model.compute_loss(user_ids_batch, movie_ids_batch, movie_ids_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        if (epoch + 1) % 2 == 0:
            print(f"   Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # 6. Evaluate
    print("6. Evaluating model...")
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        num_test_batches = 0
        for batch in test_loader:
            user_ids_batch = batch['user_id']
            movie_ids_batch = batch['item_id']
            
            # Compute test loss
            test_loss = model.compute_loss(user_ids_batch, movie_ids_batch, movie_ids_batch)
            test_losses.append(test_loss.item())
            num_test_batches += 1
    
    avg_test_loss = np.mean(test_losses)
    print(f"   Average Test Loss: {avg_test_loss:.4f}")
    
    # 7. Generate recommendations for a specific user
    print("7. Generating recommendations...")
    model.eval()
    with torch.no_grad():
        # Generate recommendations for user 0
        sample_user = torch.tensor([0])
        candidate_movies = torch.arange(0, 20)  # All 20 movies
        
        # Manual computation to debug
        print("   === Manual computation ===")
        user_embeddings = model.user_model(sample_user)
        item_embeddings = model.item_model(candidate_movies)
        print(f"   User embeddings shape: {user_embeddings.shape}")
        print(f"   Item embeddings shape: {item_embeddings.shape}")
        
        # Compute scores manually
        manual_scores = torch.matmul(user_embeddings, item_embeddings.t())
        print(f"   Manual scores shape: {manual_scores.shape}")
        print(f"   Manual scores (first 5): {manual_scores[0, :5]}")
        
        # Get top manually
        manual_top_scores, manual_top_indices = torch.topk(manual_scores, 5, dim=1)
        manual_top_items = candidate_movies[manual_top_indices]
        print(f"   Manual top items: {manual_top_items[0]}")
        print(f"   Manual top scores: {manual_top_scores[0]}")
        
        # Now use the recommend method
        print("   === Using recommend method ===")
        top_movies, top_scores = model.recommend(sample_user, candidate_movies, k=5)
        print(f"   Recommend method top items: {top_movies[0]}")
        print(f"   Recommend method top scores: {top_scores[0]}")
        
        print(f"   Top 5 recommendations for user 0:")
        for i, (movie, score) in enumerate(zip(top_movies[0], top_scores[0])):
            print(f"     {i+1}. Movie {movie.item()} (Score: {score.item():.4f})")
    
    # 8. Return results for comparison
    results = {
        'train_losses': train_losses,
        'test_loss': avg_test_loss,
        'recommendations': [(movie.item(), score.item()) for movie, score in zip(top_movies[0], top_scores[0])]
    }
    
    print("\nConsistency test completed!")
    return results


if __name__ == "__main__":
    results = run_consistency_test()
    
    # Print results in a format that could be compared
    print("\n=== RESULTS SUMMARY ===")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    print(f"Test loss: {results['test_loss']:.4f}")
    print("Top recommendations for user 0:")
    for i, (movie, score) in enumerate(results['recommendations']):
        print(f"  {i+1}. Movie {movie} (Score: {score:.4f})")