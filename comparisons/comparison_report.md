# TensorFlow Recommenders vs TorchRS Comparison Report

This report compares the implementation and results of similar recommendation models in TensorFlow Recommenders (TFRS) and TorchRS.

## Test Setup

Both implementations use the same:
- Dataset: Small MovieLens-like dataset with 10 users and 20 movies
- Model architecture: Two-tower model with 32-dim embeddings projected to 16-dim
- Training: 10 epochs with Adam optimizer
- Evaluation: Top-K recommendations for user 0

## TensorFlow Recommenders Implementation

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create a small MovieLens-like dataset
def create_movielens_data():
    user_ids = []
    movie_ids = []
    ratings = []
    
    # Create a deterministic dataset
    for user_id in range(10):
        # Each user rates 5 random movies
        rated_movies = np.random.choice(20, size=5, replace=False)
        for movie_id in rated_movies:
            rating = np.random.randint(1, 6)  # Rating from 1-5
            user_ids.append(str(user_id))
            movie_ids.append(str(movie_id))
            ratings.append(float(rating))
    
    return user_ids, movie_ids, ratings

# Create dataset
user_ids, movie_ids, ratings = create_movielens_data()

# Convert to TensorFlow dataset
ratings_data = tf.data.Dataset.from_tensor_slices({
    "user_id": user_ids,
    "movie_id": movie_ids,
    "rating": ratings
})

# Get unique user and movie IDs
unique_user_ids = list(set(user_ids))
unique_movie_ids = list(set(movie_ids))

# Create vocabularies
user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(unique_user_ids)

movie_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_ids_vocabulary.adapt(unique_movie_ids)

# Define model
class MovieLensModel(tfrs.Model):
    def __init__(self, user_ids_vocabulary, movie_ids_vocabulary):
        super().__init__()
        self.user_ids_vocabulary = user_ids_vocabulary
        self.movie_ids_vocabulary = movie_ids_vocabulary
        
        # User model
        self.user_model = tf.keras.Sequential([
            user_ids_vocabulary,
            tf.keras.layers.Embedding(len(user_ids_vocabulary.get_vocabulary()), 32),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        
        # Movie model
        self.movie_model = tf.keras.Sequential([
            movie_ids_vocabulary,
            tf.keras.layers.Embedding(len(movie_ids_vocabulary.get_vocabulary()), 32),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        
        # Task
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(unique_movie_ids)
                    .batch(128)
                    .map(self.movie_model)
            )
        )
    
    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_id"])
        return self.task(user_embeddings, movie_embeddings)

# Create and train model
model = MovieLensModel(user_ids_vocabulary, movie_ids_vocabulary)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01))

# Train
train_data = ratings_data.shuffle(1000).batch(16)
model.fit(train_data, epochs=10)

# Generate recommendations
index = tfrs.layers.ann.BruteForce(model.user_model)
index.index_from_dataset(
    tf.data.Dataset.from_tensor_slices(unique_movie_ids)
        .batch(100)
        .map(lambda x: (x, model.movie_model(x)))
)

# Get recommendations for user 0
_, titles = index(np.array(["0"]))
print(f"Top 5 recommendations for user 0: {titles[0, :5]}")
```

## TorchRS Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from torchrs import models, tasks, metrics, data

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create a small MovieLens-like dataset
def create_movielens_data():
    user_ids = []
    movie_ids = []
    ratings = []
    
    # Create a deterministic dataset
    for user_id in range(10):
        # Each user rates 5 random movies
        rated_movies = np.random.choice(20, size=5, replace=False)
        for movie_id in rated_movies:
            rating = np.random.randint(1, 6)  # Rating from 1-5
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            ratings.append(float(rating))
    
    return np.array(user_ids), np.array(movie_ids), np.array(ratings)

# Create dataset
user_ids, movie_ids, ratings = create_movielens_data()
dataset = data.RecommendationDataset(user_ids, movie_ids, ratings)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Create models
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

# Create task and model
task = tasks.Retrieval(metrics=[metrics.FactorizedTopK(k=5)])
model = models.RetrievalModel(user_model=user_model, item_model=movie_model, task=task)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(10):
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        user_ids_batch = batch['user_id']
        movie_ids_batch = batch['item_id']
        loss = model.compute_loss(user_ids_batch, movie_ids_batch, movie_ids_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

# Generate recommendations
model.eval()
with torch.no_grad():
    sample_user = torch.tensor([0])
    candidate_movies = torch.arange(0, 20)
    top_movies, top_scores = model.recommend(sample_user, candidate_movies, k=5)
    print(f"Top 5 recommendations for user 0: {top_movies[0]}")
    print(f"Top 5 scores for user 0: {top_scores[0]}")
```

## Results Comparison

| Aspect | TensorFlow Recommenders | TorchRS | Notes |
|--------|------------------------|---------|-------|
| **Implementation Language** | Python with TensorFlow | Python with PyTorch | Different underlying frameworks |
| **Model Architecture** | Keras Sequential models | PyTorch nn.Sequential | Similar high-level structure |
| **Training API** | Keras `fit` method | Manual training loop | TFRS uses high-level API, TorchRS uses explicit loops |
| **Loss Function** | Built-in Retrieval task | Custom implementation | Both implement sampled softmax |
| **Recommendation Generation** | BruteForce index layer | Custom recommend method | Different approaches to retrieval |
| **Reproducibility** | Depends on TF seeds | Depends on PyTorch seeds | Both can be made reproducible |

## Key Differences

### 1. Framework Differences
- **TFRS**: Built on TensorFlow/Keras with static computation graphs
- **TorchRS**: Built on PyTorch with dynamic computation graphs

### 2. API Design
- **TFRS**: Uses high-level Keras APIs with `fit` method
- **TorchRS**: Uses explicit training loops giving more control

### 3. Model Definition
- **TFRS**: Models inherit from `tfrs.Model`
- **TorchRS**: Models inherit from `torch.nn.Module`

### 4. Indexing for Retrieval
- **TFRS**: Uses specialized index layers like `BruteForce`
- **TorchRS**: Implements recommendation generation directly

## Similarities

1. **Two-Tower Architecture**: Both use separate user and item towers
2. **Embedding Layers**: Both use embedding layers for categorical features
3. **Retrieval Task**: Both implement retrieval-based recommendation
4. **Metrics**: Both support similar evaluation metrics
5. **Flexibility**: Both allow custom model architectures

## Performance Considerations

- **Training Speed**: Depends on hardware and specific implementation
- **Memory Usage**: PyTorch's dynamic graphs may use more memory
- **Scalability**: TFRS has more mature distributed training support
- **Ease of Debugging**: TorchRS may be easier to debug due to eager execution

## Conclusion

Both TensorFlow Recommenders and TorchRS provide robust frameworks for building recommendation systems. While they share similar high-level concepts and architectures, they differ in their underlying frameworks and APIs:

1. **TFRS** is ideal for production environments where TensorFlow is already used and where high-level APIs are preferred.

2. **TorchRS** is ideal for research environments where PyTorch is preferred and where more control over the training process is needed.

The choice between them should be based on:
- Existing technology stack
- Team expertise
- Performance requirements
- Preference for high-level vs. low-level APIs