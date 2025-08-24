# TorchRS API Reference

## Modules

### torchrs.models

#### Model
Base class for recommendation models.

#### Embedding
Embedding layer for categorical features.

#### FeatureEmbedding
Embedding layer for multiple categorical features.

#### Tower
Base tower module for two-tower architectures.

#### UserTower
User tower for two-tower recommendation models.

#### ItemTower
Item tower for two-tower recommendation models.

#### RetrievalModel
Two-tower retrieval model for recommendation systems.

### torchrs.tasks

#### Task
Base class for recommendation tasks.

#### Retrieval
Retrieval task for recommendation systems.

#### Ranking
Ranking task for recommendation systems.

### torchrs.metrics

#### FactorizedTopK
Factorized Top-K metric for retrieval evaluation.

### torchrs.data

#### RecommendationDataset
Base dataset class for recommendation systems.

#### negative_sampling
Generate negative samples for a dataset.

## Quick Examples

### Retrieval Model

```python
import torch
import torchrs as trs

# Define user and item models
user_model = torch.nn.Sequential(
    torch.nn.Embedding(num_users, 32),
    torch.nn.Linear(32, 32)
)

item_model = torch.nn.Sequential(
    torch.nn.Embedding(num_items, 32),
    torch.nn.Linear(32, 32)
)

# Define the retrieval task
task = trs.tasks.Retrieval(
    metrics=trs.metrics.FactorizedTopK(k=10)
)

# Create the full model
model = trs.models.RetrievalModel(
    user_model=user_model,
    item_model=item_model,
    task=task
)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model.compute_loss(batch['user_id'], batch['item_id'], batch['item_id'])
        loss.backward()
        optimizer.step()
```

### Ranking Model

```python
import torch
import torchrs as trs

# Define a simple ranking model
class RankingModel(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(concat).squeeze()

# Create model and task
model = RankingModel(num_users=1000, num_items=500)
task = trs.tasks.Ranking()

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        predictions = model(batch['user_id'], batch['item_id'])
        loss = task.compute_loss(predictions, batch['rating'])
        loss.backward()
        optimizer.step()
```