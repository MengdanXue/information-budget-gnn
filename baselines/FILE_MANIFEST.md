# Baseline Comparison Framework - File Manifest

## Created Files Summary

All files have been created in: `D:\Users\11919\Documents\毕业论文\paper\code\baselines\`

### Core Implementation Files

1. **`__init__.py`** (826 bytes)
   - Package initialization
   - Exports all models and data loaders
   - Version information

2. **`baseline_models.py`** (14,582 bytes)
   - Implementation of all 6 baseline models
   - ARCInspired, GAGA, CAREGNN, PCGNN, VecAugInspired, SEFraudInspired
   - Model factory function
   - Test code for validation

3. **`data_loaders.py`** (12,192 bytes)
   - Data loaders for 4 datasets
   - IEEECISLoader, YelpChiLoader, AmazonLoader, EllipticLoader
   - Unified dataset loading interface
   - Auto-download support for DGL datasets

4. **`run_baselines.py`** (11,917 bytes)
   - Main experiment script
   - Training and evaluation loops
   - Multi-seed experiment support
   - Results saving and reporting

5. **`run_quick_test.py`** (4,689 bytes)
   - Quick validation script
   - Tests all models and data loaders
   - Verifies training pipeline
   - Sanity checks before full experiments

### Dependency and Configuration

6. **`requirements_baselines.txt`** (2,813 bytes)
   - All Python dependencies
   - PyTorch, PyG, DGL requirements
   - Installation instructions
   - CUDA 12.1 support notes

### Execution Scripts

7. **`run_all_experiments.bat`** (1,612 bytes)
   - Windows batch script
   - Runs all baselines on all datasets
   - 4-stage execution pipeline

8. **`run_all_experiments.sh`** (1,625 bytes, executable)
   - Linux/Mac shell script
   - Same functionality as .bat
   - Executable permissions set

### Documentation

9. **`README.md`** (8,054 bytes)
   - Main documentation
   - Baseline method descriptions
   - Dataset information
   - Installation and usage guide
   - Citation information

10. **`EXPERIMENT_GUIDE.md`** (7,694 bytes)
    - Detailed experimental guide
    - Step-by-step instructions
    - Expected runtime and results
    - Troubleshooting tips
    - LaTeX table templates

11. **`IMPLEMENTATION_NOTES.md`** (10,016 bytes)
    - Technical implementation details
    - Model architecture comparison
    - Hyperparameter sensitivity analysis
    - Known issues and limitations
    - Validation and testing guidelines

12. **`FILE_MANIFEST.md`** (this file)
    - Complete file listing
    - File descriptions
    - Quick reference guide

## Directory Structure

```
baselines/
├── __init__.py                    # Package init
├── baseline_models.py             # Model implementations
├── data_loaders.py                # Data loaders
├── run_baselines.py               # Main experiment script
├── run_quick_test.py              # Quick validation
├── requirements_baselines.txt     # Dependencies
├── run_all_experiments.bat        # Windows batch script
├── run_all_experiments.sh         # Linux/Mac shell script
├── README.md                      # Main documentation
├── EXPERIMENT_GUIDE.md            # Detailed guide
├── IMPLEMENTATION_NOTES.md        # Technical notes
├── FILE_MANIFEST.md               # This file
└── results/                       # Results directory (auto-created)
    └── baseline_comparison_*.json
```

## File Dependencies

```
baseline_models.py
    ├─> torch
    ├─> torch_geometric
    └─> numpy

data_loaders.py
    ├─> torch
    ├─> torch_geometric
    ├─> numpy
    ├─> pandas
    ├─> pickle
    └─> dgl (optional)

run_baselines.py
    ├─> baseline_models.py
    ├─> data_loaders.py
    ├─> ../daaa_model.py (FSD-GNN models)
    ├─> torch
    └─> sklearn

run_quick_test.py
    ├─> baseline_models.py
    ├─> data_loaders.py
    └─> run_baselines.py
```

## Usage Quick Reference

### Installation
```bash
pip install -r requirements_baselines.txt
```

### Quick Test
```bash
python run_quick_test.py
```

### Run Single Experiment
```bash
python run_baselines.py --model ARC --dataset ieee-cis --seeds 42 123 456
```

### Run Full Comparison
```bash
# Windows
run_all_experiments.bat

# Linux/Mac
bash run_all_experiments.sh
```

## File Sizes

| File | Size | Type |
|------|------|------|
| `__init__.py` | 826 B | Python |
| `baseline_models.py` | 14.6 KB | Python |
| `data_loaders.py` | 12.2 KB | Python |
| `run_baselines.py` | 11.9 KB | Python |
| `run_quick_test.py` | 4.7 KB | Python |
| `requirements_baselines.txt` | 2.8 KB | Text |
| `run_all_experiments.bat` | 1.6 KB | Batch |
| `run_all_experiments.sh` | 1.6 KB | Shell |
| `README.md` | 8.1 KB | Markdown |
| `EXPERIMENT_GUIDE.md` | 7.7 KB | Markdown |
| `IMPLEMENTATION_NOTES.md` | 10.0 KB | Markdown |
| `FILE_MANIFEST.md` | (this file) | Markdown |
| **Total** | **~76 KB** | |

## Code Statistics

### Lines of Code
- **Python Code:** ~1,200 lines
- **Documentation:** ~800 lines
- **Comments:** ~300 lines
- **Total:** ~2,300 lines

### Model Implementations
- 6 baseline models
- 4 data loaders
- 1 unified training framework
- Multiple evaluation metrics

### Supported Configurations
- 6 baseline methods (+ 4 FSD-GNN variants)
- 4 datasets
- 10 default random seeds
- Flexible hyperparameters

## Checksum Information

For verification purposes, you can compute checksums:

```bash
# MD5 checksums
md5sum *.py *.txt *.md

# SHA256 checksums
sha256sum *.py *.txt *.md
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-23 | Initial release |
|  |  | - All 6 baselines implemented |
|  |  | - 4 dataset loaders |
|  |  | - Complete documentation |

## External Dependencies

### Required
- PyTorch >= 2.1.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.21.0
- Scikit-learn >= 1.0.0

### Optional
- DGL >= 1.0.0 (for YelpChi/Amazon auto-download)
- CUDA 12.1 (for GPU acceleration)
- WandB (for experiment tracking)

## Data Requirements

### Expected Data Locations
- IEEE-CIS: `../processed/ieee_cis_graph.pkl`
- Elliptic: `../data/elliptic_weber_split.pkl`
- YelpChi: Auto-download or `../data/yelpchi_processed.pkl`
- Amazon: Auto-download or `../data/amazon_processed.pkl`

### Data Format
All datasets use PyTorch Geometric `Data` format:
- Node features: `[N, F]`
- Edge index: `[2, E]`
- Labels: `[N]`
- Train/Val/Test masks: `[N]`

## Output Files

### Generated During Execution
```
results/
├── baseline_comparison_20241223_001234.json
├── baseline_comparison_20241223_005678.json
└── ...
```

### Result File Format
- JSON format
- Nested structure: dataset -> model -> metrics
- Contains mean, std, and raw results
- Timestamped filenames

## Notes for Paper

### What to Include in Paper
1. **Method comparison table** (from README.md)
2. **Performance results** (from experiment outputs)
3. **Implementation details** (from IMPLEMENTATION_NOTES.md)
4. **Citation information** (from README.md)

### What to Include in Supplementary
1. Complete implementation code
2. Hyperparameter sensitivity analysis
3. Additional ablation studies
4. Detailed experimental setup

## Maintenance

### Future Updates
- [ ] Add more baseline methods (e.g., GDN, FRAUDRE)
- [ ] Support mini-batch training for large graphs
- [ ] Add distributed training support
- [ ] Implement automatic hyperparameter tuning
- [ ] Add visualization tools for results

### Known Issues
- VecAug and SEFraud are approximations (no official code)
- Large graphs may require mini-batch training
- GPU memory requirements vary by model

## Contact

For questions, issues, or contributions:
- Open an issue in the repository
- Contact paper authors
- Refer to individual baseline repositories

---

**Total Framework Size:** ~76 KB (code + docs)
**Lines of Code:** ~2,300 lines
**Implementation Time:** Research + coding
**Status:** Ready for experiments

Last updated: 2024-12-23
