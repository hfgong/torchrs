# Contributing to TorchRS

We welcome contributions to TorchRS! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/torchrs.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

1. Install the package in development mode:
   ```bash
   cd torchrs
   pip install -e .
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Style

We follow the PEP 8 style guide for Python code. Please ensure your code adheres to these standards.

### Formatting
- Use 4 spaces for indentation (no tabs)
- Limit lines to 88 characters (to match Black's default)
- Use meaningful variable and function names
- Write docstrings for all public classes and functions

### Tools
We use Black for code formatting and Flake8 for linting:
```bash
black .
flake8 .
```

## Testing

All contributions should include appropriate tests. We use pytest for testing.

### Running Tests
```bash
cd torchrs
python tests/run_tests.py
```

### Writing Tests
- Place tests in the appropriate file in the `tests/` directory
- Use descriptive test function names
- Test both expected behavior and edge cases
- Keep tests focused and independent

## Documentation

### Docstrings
Use Google-style docstrings for all public classes and functions:

```python
def example_function(param1: int, param2: str) -> bool:
    """Example function with types documented in the docstring.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.
    """
    return True
```

### API Documentation
Update the API documentation in `docs/api.md` when adding new public APIs.

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation as needed
3. Add an entry to the changelog (if applicable)
4. Submit a pull request with a clear title and description
5. Address any feedback from maintainers

## Reporting Issues

Please use the GitHub issue tracker to report bugs or suggest features. When reporting a bug, include:

1. A clear description of the problem
2. Steps to reproduce the issue
3. Expected vs. actual behavior
4. Environment information (OS, Python version, etc.)

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## Questions?

If you have any questions about contributing, feel free to open an issue or contact the maintainers.