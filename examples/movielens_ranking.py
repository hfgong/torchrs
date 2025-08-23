"""
MovieLens Ranking Example

This example demonstrates how to build a ranking model using TorchRS
with the MovieLens dataset.
"""

import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, data


def create_sample_data_with_ratings():
    """Create sample MovieLens-like data with ratings for demonstration."""
    # Generate sample user-item interactions with ratings
    num_users = 1000
    num_items = 500
    num_interactions = 10000
    
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    ratings = np.random.randint(1, 6, num_interactions)  # Ratings from 1-5
    
    return user_ids, item_ids, ratings


class RankingModel(nn.Module):
    """Simple ranking model that predicts ratings."""
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(concat).squeeze()


def main():
    # Create sample data
    user_ids, item_ids, ratings = create_sample_data_with_ratings()
    
    # Create dataset
    dataset = data.RecommendationDataset(user_ids, item_ids, ratings)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = RankingModel(num_users=1000, num_items=500)
    
    # Define ranking task
    task = tasks.Ranking()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Training loop
    model.train()
    for epoch in range(99):  # Train for 10 epochs
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            user_ids = batch['user_id']
            item_ids = batch['item_id']
            ratings = batch['rating']
            
            # Forward pass
            predictions = model(user_ids, item_ids)
            
            # Compute loss
            loss = task.compute_loss(predictions, ratings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    # Example of making predictions
    model.eval()
    with torch.no_grad():
        # Predict ratings for sample user-item pairs
        sample_users = torch.tensor([42, 100, 200])
        sample_items = torch.tensor([10, 50, 100])
        
        predictions = model(sample_users, sample_items)
        print(f"\nPredicted ratings:")
        for user, item, rating in zip(sample_users, sample_items, predictions):
            print(f"  User {user.item()}, Item {item.item()}: {rating.item():.2f}")


if __name__ == "__main__":
    main()