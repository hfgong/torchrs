# Migration Guide: TensorFlow Recommenders to TorchRS

This guide helps you migrate from TensorFlow Recommenders (TFRS) to TorchRS, highlighting the key differences and similarities between the two libraries.

## Key Differences

### Framework
- **TFRS**: Built on TensorFlow/Keras
- **TorchRS**: Built on PyTorch

### Computation Graph
- **TFRS**: Static computation graph (by default)
- **TorchRS**: Dynamic computation graph (eager execution)

### Model Definition
- **TFRS**: Models inherit from `tfrs.Model`
- **TorchRS**: Models inherit from `torch.nn.Module` or `torchrs.models.Model`

## Migration Examples

### Retrieval Model

#### TFRS Version
```python
import tensorflow as tf
import tensorflow_recommenders as tfrs

class MovieLensModel(tfrs.Model):
  def __init__(self, user_model, movie_model):
    super().__init__()
    self.user_model = user_model
    self.movie_model = movie_model
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.movie_model)
        )
    )

  def compute_loss(self, features, training=False):
    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])
    return self.task(user_embeddings, movie_embeddings)
```

#### TorchRS Version
```python
import torch
import torchrs as trs

class MovieLensModel(trs.models.RetrievalModel):
  def __init__(self, user_model, movie_model):
    task = trs.tasks.Retrieval(
        metrics=[trs.metrics.FactorizedTopK(k=10)]
    )
    super().__init__(
        user_model=user_model,
        item_model=movie_model,
        task=task
    )
```

### Data Pipeline

#### TFRS Version
```python
import tensorflow_datasets as tfds

ratings = tfds.load("movielens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)
```

#### TorchRS Version
```python
import torch
from torchrs.data import RecommendationDataset

# Assuming you have user_ids and item_ids as lists/arrays
dataset = RecommendationDataset(user_ids, item_ids)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Training Loop

#### TFRS Version
```python
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
cached_train = train.shuffle(100_000).batch(8192).cache()
model.fit(cached_train, epochs=3)
```

#### TorchRS Version
```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model.compute_loss(batch['user_id'], batch['item_id'], batch['item_id'])
        loss.backward()
        optimizer.step()
```

## Component Mapping

| TFRS Component | TorchRS Equivalent | Notes |
|----------------|--------------------|-------|
| `tfrs.Model` | `torchrs.models.Model` | Base model class |
| `tfrs.tasks.Retrieval` | `torchrs.tasks.Retrieval` | Retrieval task |
| `tfrs.tasks.Ranking` | `torchrs.tasks.Ranking` | Ranking task |
| `tfrs.metrics.FactorizedTopK` | `torchrs.metrics.FactorizedTopK` | Top-K metric |
| `tf.keras.layers.Embedding` | `torch.nn.Embedding` or `torchrs.models.Embedding` | Embedding layer |

## Best Practices

### Model Design
- In TFRS, you typically override `compute_loss`; in TorchRS, you can use the built-in `compute_loss` method or override it if needed.
- TorchRS models are more flexible due to PyTorch's dynamic nature.

### Data Handling
- TFRS integrates deeply with TensorFlow Datasets; TorchRS works with PyTorch DataLoaders.
- For complex data pipelines, you may need to rewrite data preprocessing logic.

### Training
- TFRS uses Keras' high-level training API; TorchRS uses standard PyTorch training loops.
- This gives more control in TorchRS but requires more boilerplate code.

## Performance Considerations

- **GPU Utilization**: Both libraries support GPU acceleration, but the implementation details differ.
- **Distributed Training**: TFRS leverages TensorFlow's distribution strategies; TorchRS can use PyTorch's distributed training capabilities.
- **Memory Management**: PyTorch's dynamic graph can sometimes use more memory than TensorFlow's static graph, but this is often offset by better memory optimization techniques.

## When to Migrate

Consider migrating to TorchRS if you:
1. Prefer PyTorch's ecosystem and flexibility
2. Need dynamic computation graphs for research purposes
3. Want better debugging capabilities
4. Are already using PyTorch in your stack
5. Require more control over the training process

## Conclusion

Migrating from TFRS to TorchRS involves understanding the differences in API design and framework philosophy. While the core concepts remain the same, the implementation details differ due to the underlying frameworks. The migration is generally straightforward for most use cases, especially with the similar high-level abstractions provided by both libraries.