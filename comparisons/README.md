# TorchRS vs TensorFlow Recommenders Comparisons

This directory contains code and documentation comparing TorchRS with TensorFlow Recommenders (TFRS).

## Contents

### Comparison Reports
- `comparison_report.md` - Initial comparison between TFRS and TorchRS
- `final_comparison_report.md` - Comprehensive comparison with improved TorchRS implementation
- `user_recommendation_comparison.md` - Detailed analysis of user recommendation similarities

### Comparison Code
- `tfrs_comparison.py` - TensorFlow Recommenders implementation for comparison
- `tfrs_comparison_simple.py` - Simplified TFRS implementation
- `simple_tfrs_test.py` - Basic TFRS test
- `movielens_comparison.py` - Comparison using MovieLens dataset
- `simple_movielens_comparison.py` - Simplified MovieLens data loader

## Purpose

These files were created to:
1. Validate that TorchRS produces comparable results to TFRS
2. Demonstrate the similarities and differences between the frameworks
3. Show that both frameworks can generate similar recommendations for the same users
4. Provide examples of equivalent implementations in both frameworks

## Usage

To run the comparison code:
```bash
# Run TFRS examples (requires TensorFlow and TFRS)
python tfrs_comparison.py

# Run TorchRS examples (requires TorchRS library)
python ../examples/run_examples.py
```

Note: Some comparison code may require additional dependencies or have specific requirements.