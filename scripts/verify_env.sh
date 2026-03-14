#!/usr/bin/env bash
# verify_env.sh — Verify the vuln-prioritize conda environment
set -euo pipefail

echo "=== Environment Verification ==="

# Check conda env
if [[ "${CONDA_DEFAULT_ENV:-}" != "vuln-prioritize" ]]; then
    echo "WARNING: Expected conda env 'vuln-prioritize', got '${CONDA_DEFAULT_ENV:-none}'"
    echo "Run: conda activate vuln-prioritize"
fi

# Check Python version
echo -n "Python: "
python3 --version

# Check key packages
echo ""
echo "Package versions:"
python3 -c "
import importlib
packages = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('sklearn', 'scikit-learn'),
    ('xgboost', 'xgboost'),
    ('shap', 'shap'),
    ('requests', 'requests'),
    ('matplotlib', 'matplotlib'),
    ('tqdm', 'tqdm'),
    ('pytest', 'pytest'),
]
for module, name in packages:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, '__version__', 'installed')
        print(f'  {name}: {ver}')
    except ImportError:
        print(f'  {name}: NOT INSTALLED')
        exit(1)
print()
print('All packages verified.')
"

echo ""
echo "=== Disk Space ==="
df -h /home/azureuser/ | tail -1

echo ""
echo "=== Environment OK ==="
