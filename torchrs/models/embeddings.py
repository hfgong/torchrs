import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Embedding layer for recommendation systems.
    
    This layer creates embeddings for categorical features commonly
    found in recommendation systems such as user IDs, item IDs, 
    and other categorical features.
    """
    
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        """Initialize the embedding layer.
        
        Args:
            num_embeddings: Size of the vocabulary (number of unique items)
            embedding_dim: Dimension of the embedding vectors
            padding_idx: Index to use for padding (optional)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
    def forward(self, indices):
        """Forward pass through the embedding layer.
        
        Args:
            indices: Tensor of indices to embed
            
        Returns:
            Embedded representations of the input indices
        """
        return self.embedding(indices)
    
    
class FeatureEmbedding(nn.Module):
    """Embedding layer for multiple categorical features.
    
    This layer creates embeddings for multiple categorical features
    and combines them into a single representation.
    """
    
    def __init__(self, feature_sizes, embedding_dim):
        """Initialize the feature embedding layer.
        
        Args:
            feature_sizes: List of sizes for each categorical feature
            embedding_dim: Dimension of the embedding vectors
        """
        super().__init__()
        self.feature_sizes = feature_sizes
        self.embedding_dim = embedding_dim
        
        # Create embedding layers for each feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim) for size in feature_sizes
        ])
        
    def forward(self, features):
        """Forward pass through the feature embedding layer.
        
        Args:
            features: List of tensors, each containing indices for a feature
            
        Returns:
            Combined embedded representation of all features
        """
        # Embed each feature
        embedded_features = [
            embedding(feature) for embedding, feature in zip(self.embeddings, features)
        ]
        
        # Sum embeddings to create combined representation
        combined = torch.sum(torch.stack(embedded_features), dim=0)
        
        return combined