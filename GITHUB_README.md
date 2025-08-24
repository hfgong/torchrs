# TorchRS - PyTorch Recommendation Systems Library

A PyTorch-based library for building recommendation systems, inspired by TensorFlow Recommenders (TFRS).

## Repository Status

This repository has been successfully pushed to GitHub at: https://github.com/hfgong/torchrs

## What's Included

### Core Library
- Complete implementation of recommendation system components in PyTorch
- Models module with embeddings, towers, and retrieval models
- Tasks module with retrieval and ranking tasks
- Metrics module with FactorizedTopK and other evaluation metrics
- Data module with recommendation datasets and utilities

### Documentation
- Comprehensive code documentation with detailed docstrings
- API reference and usage examples
- Migration guide from TensorFlow Recommenders
- Contributing guidelines and license information

### Examples
- MovieLens retrieval example
- MovieLens ranking example
- Complete end-to-end implementations

### Testing
- Unit tests for all components
- End-to-end integration tests
- Consistency tests with detailed validation

### Comparisons
- Detailed comparison with TensorFlow Recommenders
- User recommendation analysis
- Equivalent implementations in both frameworks

## Key Features

1. **Two-Tower Architecture**: Implements scalable retrieval models with separate user and item towers
2. **Flexible Design**: Modular components that can be easily customized and extended
3. **Comprehensive API**: High-level interfaces for common recommendation tasks
4. **Proper Loss Functions**: Sampled softmax and other appropriate loss functions for recommendation tasks
5. **Evaluation Metrics**: Built-in metrics for assessing recommendation quality
6. **Well-Documented**: Extensive documentation explaining concepts and usage

## Installation

```bash
# Clone the repository
git clone https://github.com/hfgong/torchrs.git

# Navigate to the directory
cd torchrs

# Install in development mode
pip install -e .
```

## Usage

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
# ... (training loop)
```

## Repository Structure

```
torchrs/
├── torchrs/              # Main library code
│   ├── data/            # Data handling utilities
│   ├── metrics/         # Evaluation metrics
│   ├── models/          # Model components
│   ├── tasks/           # Task definitions
│   └── __init__.py      # Package initialization
├── examples/            # Usage examples
├── tests/               # Test suite
├── comparisons/         # TFRS vs TorchRS comparisons
├── docs/                # Documentation
├── README.md            # Main README
├── setup.py             # Installation configuration
└── requirements.txt     # Dependencies
```

## Contributing

Contributions are welcome! Please see the [contributing guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.