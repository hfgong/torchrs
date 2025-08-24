# TorchRS: PyTorch Recommendation Systems Library

A PyTorch-based library for building recommendation systems, inspired by TensorFlow Recommenders (TFRS).

## Overview

TorchRS is a library for building recommendation systems using PyTorch. It provides high-level APIs for common recommendation tasks like retrieval and ranking, while leveraging PyTorch's flexibility and ease of use.

## Features

- **Retrieval Models**: Build two-tower models for candidate selection
- **Ranking Models**: Create models for scoring and ranking items
- **Pre-built Components**: Ready-to-use layers, metrics, and loss functions
- **Flexible Architecture**: Easily customize every aspect of your models
- **PyTorch Native**: Fully integrated with PyTorch ecosystem
- **Comprehensive Documentation**: Well-documented code with detailed explanations

## Installation

```bash
pip install torchrs
```

## Quick Start

Here's how to build a simple retrieval model using TorchRS:

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
    metrics=trs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(model.candidate_model)
    )
)

# Create the full model
model = trs.models.RetrievalModel(
    user_model=user_model,
    item_model=item_model,
    task=task
)

# Train the model
model.fit(training_data, epochs=5)
```

## Running Examples

To run the provided examples:

```bash
cd torchrs
python examples/run_examples.py
```

This will run both the MovieLens retrieval and ranking examples.

## Running Tests

To run the tests:

```bash
cd torchrs
python tests/run_tests.py
```

## Code Documentation

The library includes comprehensive code documentation with detailed explanations of:
- All classes, methods, and functions
- Parameters, return values, and exceptions
- Recommendation system concepts and mathematical foundations
- Practical usage patterns and best practices

See our [code documentation improvements](torchrs/code_documentation_improvements.md) for details.

## Comparison with TensorFlow Recommenders

We've conducted a detailed comparison between TorchRS and TensorFlow Recommenders. See our [comparison reports](comparisons/) for more details.

## Documentation

- [API Reference](docs/api.md)
- [Tutorials](docs/tutorials/)
- [Migration Guide from TFRS](docs/migration.md)

## Examples

Check out our [examples](examples/) directory for complete implementations:

- [MovieLens Retrieval](examples/movielens_retrieval.py)
- [MovieLens Ranking](examples/movielens_ranking.py)
- [Custom Model](examples/custom_model.py)

## Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for more details.

## License

[Apache License 2.0](LICENSE)