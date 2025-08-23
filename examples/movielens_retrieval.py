"""
MovieLens Retrieval Example

This example demonstrates how to build a retrieval model using TorchRS
with the MovieLens dataset.
"""

import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data


def create_sample_data():
    """Create sample MovieLens-like data for demonstration."""
    # Generate sample user-item interactions
    num_users = 1000
    num_items = 500
    num_interactions = 10000
    
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    
    return user_ids, item_ids


def main():
    # Create sample data
    user_ids, item_ids = create_sample_data()
    
    # Create dataset
    dataset = data.RecommendationDataset(user_ids, item_ids)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define user and item models
    user_model = nn.Sequential(
        nn.Embedding(1000, 64),  # User embedding
        nn.Linear(64, 32),       # Projection layer
        nn.ReLU()
    )
    
    item_model = nn.Sequential(
        nn.Embedding(500, 64),   # Item embedding
        nn.Linear(64, 32),       # Projection layer
        nn.ReLU()
    )
    
    # Define retrieval task with metrics
    task = tasks.Retrieval(
        metrics=[metrics.FactorizedTopK(k=10)]
    )
    
    # Create retrieval model
    model = models.RetrievalModel(
        user_model=user_model,
        item_model=item_model,
        task=task
    )
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            user_ids = batch['user_id']
            item_ids = batch['item_id']
            
            # Compute loss
            loss = model.compute_loss(user_ids, item_ids, item_ids)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    # Example of generating recommendations
    model.eval()
    with torch.no_grad():
        # Generate recommendations for a sample user
        sample_user = torch.tensor([42])  # User ID 42
        candidate_items = torch.arange(0, 100)  # First 100 items as candidates
        
        top_items, top_scores = model.recommend(sample_user, candidate_items, k=5)
        print(f"\nTop 5 recommendations for user 42:")
        for i, (item, score) in enumerate(zip(top_items[0], top_scores[0])):
            print(f"  {i+1}. Item {item.item()} (Score: {score.item():.4f})")


if __name__ == "__main__":
    main()