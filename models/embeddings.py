import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Embedding layer for recommendation systems.
    
    This layer creates dense vector representations (embeddings) for categorical
    features commonly found in recommendation systems such as user IDs, item IDs,
    and other categorical features. Embeddings are learned during training and
    capture semantic relationships between categorical values.
    
    The embedding dimension is a hyperparameter that controls the size of the
    learned representations. Larger dimensions can capture more complex relationships
    but may lead to overfitting and increased computational cost.
    """
    
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        """Initialize the embedding layer.
        
        Args:
            num_embeddings (int): Size of the vocabulary (number of unique items).
                For example, if you have 1000 unique users, num_embeddings=1000.
            embedding_dim (int): Dimension of the embedding vectors.
                This determines the size of the learned representations.
            padding_idx (int, optional): Index to use for padding. If specified,
                the embedding at this index will not be updated during training
                and will remain zero. This is useful for handling variable-length
                sequences where padding is used.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Create the underlying PyTorch embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
    def forward(self, indices):
        """Forward pass through the embedding layer.
        
        Takes indices of categorical items and returns their dense vector
        representations. Each index is mapped to its corresponding embedding
        vector.
        
        Args:
            indices (torch.Tensor): Tensor of indices to embed. Shape should be
                (batch_size,) or (batch_size, sequence_length) for sequential data.
                
        Returns:
            torch.Tensor: Embedded representations of the input indices.
                Shape will be (batch_size, embedding_dim) or 
                (batch_size, sequence_length, embedding_dim).
        """
        return self.embedding(indices)
    
    
class FeatureEmbedding(nn.Module):
    """Embedding layer for multiple categorical features.
    
    This layer creates embeddings for multiple categorical features and combines
    them into a single representation. This is useful when you have multiple
    categorical features for users or items (e.g., user age group, gender, and
    location; or item category, brand, and price range).
    
    Each feature gets its own embedding layer, and the embeddings are summed
    to create a combined representation. This approach assumes that all features
    contribute equally to the final representation.
    """
    
    def __init__(self, feature_sizes, embedding_dim):
        """Initialize the feature embedding layer.
        
        Args:
            feature_sizes (list of int): List of sizes for each categorical feature.
                For example, if you have features with 10, 5, and 20 unique values
                respectively, feature_sizes=[10, 5, 20].
            embedding_dim (int): Dimension of the embedding vectors. All features
                will use the same embedding dimension for consistent combination.
        """
        super().__init__()
        self.feature_sizes = feature_sizes
        self.embedding_dim = embedding_dim
        
        # Create separate embedding layers for each feature
        # Each feature gets its own embedding table
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim) for size in feature_sizes
        ])
        
    def forward(self, features):
        """Forward pass through the feature embedding layer.
        
        Takes multiple feature tensors and returns their combined embedding
        representation. Each feature is embedded separately and then summed.
        
        Args:
            features (list of torch.Tensor): List of tensors, each containing
                indices for a different feature. All tensors should have the
                same batch size.
                
        Returns:
            torch.Tensor: Combined embedded representation of all features.
                Shape will be (batch_size, embedding_dim).
        """
        # Embed each feature separately
        # features[i] has shape (batch_size,)
        # embedded_features[i] will have shape (batch_size, embedding_dim)
        embedded_features = [
            embedding(feature) for embedding, feature in zip(self.embeddings, features)
        ]
        
        # Sum embeddings to create combined representation
        # This assumes all features contribute equally to the final representation
        combined = torch.sum(torch.stack(embedded_features), dim=0)
        
        return combined