#!/bin/bash
# Verification script for TorchRS installation

echo "Verifying TorchRS installation..."
echo "================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Please run this script from the torchrs root directory."
    exit 1
fi

# Install the package
echo "Installing TorchRS in development mode..."
pip install -e .

# Test import
echo "Testing import..."
python -c "import torchrs; print('✓ TorchRS imported successfully')"

# Test main components
echo "Testing main components..."
python -c "
import torchrs as trs
components = {
    'Models': hasattr(trs, 'models'),
    'Tasks': hasattr(trs, 'tasks'),
    'Metrics': hasattr(trs, 'metrics'),
    'Data': hasattr(trs, 'data')
}
for name, available in components.items():
    status = '✓' if available else '✗'
    print(f'{status} {name}: {available}')
"

# Test key classes
echo "Testing key classes..."
python -c "
import torchrs as trs
classes = [
    ('models.RetrievalModel', hasattr(trs.models, 'RetrievalModel')),
    ('tasks.Retrieval', hasattr(trs.tasks, 'Retrieval')),
    ('tasks.Ranking', hasattr(trs.tasks, 'Ranking')),
    ('metrics.FactorizedTopK', hasattr(trs.metrics, 'FactorizedTopK')),
    ('data.RecommendationDataset', hasattr(trs.data, 'RecommendationDataset'))
]
for name, available in classes:
    status = '✓' if available else '✗'
    print(f'{status} {name}: {available}')
"

echo ""
echo "Installation verification complete!"
echo "TorchRS is ready to use."