# NAA Ablation Study - File Index

## Quick Navigation

| Need to...? | File to Use |
|-------------|-------------|
| **Get started quickly** | [`ABLATION_QUICK_START.md`](#ablation_quick_startmd) |
| **Understand the framework** | [`ABLATION_STUDY_SUMMARY.md`](#ablation_study_summarymd) |
| **Learn detailed usage** | [`ABLATION_STUDY_README.md`](#ablation_study_readmemd) |
| **Run experiments** | [`ablation_study.py`](#ablation_studypy) |
| **Test setup** | [`test_ablation_setup.py`](#test_ablation_setuppy) |
| **Automate all datasets** | [`run_ablation_study.bat`](#run_ablation_studybat) or [`.sh`](#run_ablation_studysh) |

## File Descriptions

### 📘 Documentation Files

#### `ABLATION_QUICK_START.md`
**Purpose**: Get up and running in < 5 minutes

**Contents**:
- TL;DR commands for running experiments
- Basic usage examples
- Output file descriptions
- Troubleshooting quick fixes

**Use when**: You want to start immediately without reading details.

**Size**: ~3 pages

---

#### `ABLATION_STUDY_README.md`
**Purpose**: Comprehensive reference guide

**Contents**:
- Detailed NAA component explanations
- Full ablation experiment descriptions
- Advanced usage and customization
- Expected results with interpretation
- Integration guide for paper
- Troubleshooting guide

**Use when**: You need complete information about the framework.

**Size**: ~15 pages

---

#### `ABLATION_STUDY_SUMMARY.md`
**Purpose**: High-level overview and workflow

**Contents**:
- Package overview
- NAA mechanism summary
- Experiment descriptions
- Expected results preview
- Paper integration checklist
- Key insights for writing

**Use when**: You want to understand what the package does before diving in.

**Size**: ~8 pages

---

#### `ABLATION_INDEX.md` (This File)
**Purpose**: Navigation and file organization

**Contents**:
- Quick navigation table
- File descriptions with use cases
- Recommended reading order
- File dependencies

**Use when**: You need to find the right file for your task.

---

### 💻 Code Files

#### `ablation_study.py`
**Purpose**: Main ablation study implementation

**What it does**:
- Implements NAA mechanism variants
- Runs component ablation experiments
- Performs lambda sensitivity analysis
- Computes statistical significance
- Generates LaTeX tables
- Saves results to JSON

**Key functions**:
- `NAA_Full` - Full NAA model with configurable components
- `BaselineGCN` - Baseline without NAA
- `run_ablation_experiment()` - Main experiment runner
- `generate_ablation_table()` - LaTeX table generation
- `compute_statistical_significance()` - Statistical tests

**Usage**:
```bash
python ablation_study.py \
    --data_path processed/elliptic_graph.pkl \
    --dataset_name "Elliptic" \
    --output_dir ablation_results/elliptic
```

**Lines of code**: ~700

---

#### `test_ablation_setup.py`
**Purpose**: Verify environment is correctly configured

**What it does**:
- Checks Python packages are installed
- Verifies data files exist
- Tests model imports
- Tests model instantiation
- Tests forward pass
- Runs mini training (1 epoch)

**Usage**:
```bash
python test_ablation_setup.py
```

**Expected output**: "SUCCESS: All tests passed!"

**Lines of code**: ~350

---

#### `run_ablation_study.bat`
**Purpose**: Windows automation script

**What it does**:
- Checks Python installation
- Runs ablation on all available datasets
- Handles errors gracefully
- Displays progress
- Prints summary

**Usage**:
```bash
run_ablation_study.bat
```

**Datasets tested**:
- Elliptic
- IEEE-CIS
- YelpChi
- Amazon

**Runtime**: ~4-5 hours (5 seeds) or ~8-10 hours (10 seeds)

---

#### `run_ablation_study.sh`
**Purpose**: Linux/Mac automation script

**What it does**: Same as `.bat` file, but for Unix systems

**Usage**:
```bash
bash run_ablation_study.sh
```

**Note**: May need to make executable first:
```bash
chmod +x run_ablation_study.sh
```

---

### 📊 Output Files

After running ablation study, each dataset gets a directory with three files:

#### `ablation_results/{dataset}/ablation_table.tex`
**Purpose**: LaTeX table for paper

**Contents**:
```latex
\begin{table}[t]
\caption{Ablation Study: NAA Component Analysis on {Dataset}}
...
\end{table}
```

**Use**: Copy directly into paper LaTeX

---

#### `ablation_results/{dataset}/lambda_sensitivity_table.tex`
**Purpose**: Lambda sensitivity LaTeX table

**Contents**:
```latex
\begin{table}[t]
\caption{Sensitivity Analysis: Impact of $\lambda$ on {Dataset}}
...
\end{table}
```

**Use**: Include in paper or supplementary material

---

#### `ablation_results/{dataset}/ablation_results.json`
**Purpose**: Complete results in JSON format

**Contents**:
```json
{
  "dataset": "Elliptic",
  "seeds": [...],
  "ablation": {...},
  "lambda_sensitivity": {...},
  "significance_tests": {...}
}
```

**Use**: Further analysis, plotting, or archival

---

## Recommended Reading Order

### For First-Time Users

1. **Start**: `ABLATION_QUICK_START.md` (5 min)
   - Get basic understanding
   - See example commands

2. **Test**: Run `test_ablation_setup.py` (2 min)
   - Verify setup works
   - Catch issues early

3. **Run**: `run_ablation_study.bat` or `.sh` (4-8 hours)
   - Let it run in background
   - Check progress occasionally

4. **Review**: Output files in `ablation_results/` (10 min)
   - Look at LaTeX tables
   - Check JSON for details

5. **Learn**: `ABLATION_STUDY_README.md` (20 min)
   - Understand components
   - Interpret results
   - Write paper section

### For Paper Writing

1. **Overview**: `ABLATION_STUDY_SUMMARY.md`
   - Get key insights
   - Review expected findings

2. **Results**: Check `ablation_results/{dataset}/*.tex`
   - Copy tables to paper
   - Note significant findings

3. **Writing**: Use templates in `ABLATION_STUDY_README.md`
   - Section 5.3 structure
   - Example text
   - Interpretation guide

### For Troubleshooting

1. **Quick fixes**: `ABLATION_QUICK_START.md` → Troubleshooting section
2. **Detailed help**: `ABLATION_STUDY_README.md` → Troubleshooting section
3. **Test issues**: Run `test_ablation_setup.py` to diagnose

### For Customization

1. **Understanding**: `ABLATION_STUDY_README.md` → Advanced usage
2. **Code**: Read `ablation_study.py` comments
3. **Modify**: Edit configurations in Python API (see README)

---

## File Dependencies

```
DOCUMENTATION (no dependencies)
├── ABLATION_INDEX.md (this file)
├── ABLATION_QUICK_START.md
├── ABLATION_STUDY_README.md
└── ABLATION_STUDY_SUMMARY.md

CODE FILES
├── ablation_study.py
│   └── Requires: torch, torch_geometric, sklearn, scipy, numpy
│
├── test_ablation_setup.py
│   └── Imports: ablation_study.py
│
├── run_ablation_study.bat
│   └── Calls: ablation_study.py
│
└── run_ablation_study.sh
    └── Calls: ablation_study.py

DATA (required for running)
└── processed/
    ├── elliptic_graph.pkl
    ├── ieee_cis_graph.pkl
    ├── yelpchi_graph.pkl
    └── amazon_graph.pkl

OUTPUT (generated after running)
└── ablation_results/
    └── {dataset}/
        ├── ablation_table.tex
        ├── lambda_sensitivity_table.tex
        └── ablation_results.json
```

---

## Quick Command Reference

### Setup & Testing
```bash
# Check setup
python test_ablation_setup.py

# Check Python packages
pip list | grep torch
```

### Running Experiments
```bash
# Single dataset
python ablation_study.py --data_path processed/elliptic_graph.pkl --dataset_name Elliptic

# All datasets (Windows)
run_ablation_study.bat

# All datasets (Linux/Mac)
bash run_ablation_study.sh

# Custom seeds
python ablation_study.py --seeds 42 123 456 789 1024 --data_path processed/elliptic_graph.pkl --dataset_name Elliptic

# CPU only
python ablation_study.py --device cpu --data_path processed/elliptic_graph.pkl --dataset_name Elliptic
```

### Checking Results
```bash
# View JSON results
cat ablation_results/elliptic/ablation_results.json

# Or with Python
python -m json.tool ablation_results/elliptic/ablation_results.json

# Check LaTeX tables
cat ablation_results/elliptic/ablation_table.tex
```

---

## File Sizes & Metrics

| File | Type | Lines | Size | Reading Time |
|------|------|-------|------|--------------|
| `ablation_study.py` | Code | ~700 | ~25 KB | - |
| `test_ablation_setup.py` | Code | ~350 | ~12 KB | - |
| `run_ablation_study.bat` | Script | ~150 | ~4 KB | - |
| `run_ablation_study.sh` | Script | ~130 | ~3 KB | - |
| `ABLATION_QUICK_START.md` | Docs | ~200 | ~8 KB | 5 min |
| `ABLATION_STUDY_README.md` | Docs | ~700 | ~30 KB | 20 min |
| `ABLATION_STUDY_SUMMARY.md` | Docs | ~400 | ~18 KB | 10 min |
| `ABLATION_INDEX.md` | Docs | ~250 | ~10 KB | 5 min |

**Total package size**: ~110 KB (code + docs)

---

## Integration with Other Files

### Related Model Files
- `daaa_model.py` - Contains DAAA models that use NAA components
- `train_ieee_cis.py` - Training script template

### Related Ablation Studies
- `ablation_delta_agg.py` - Ablation for δ_agg metric
- `stratified_analysis.py` - Degree-stratified analysis

### Related Utilities
- `generate_latex_tables.py` - General LaTeX table generation
- `statistical_analysis.py` - Statistical testing

### Data Processing
- `build_multi_graphs.py` - Graph construction
- `ieee_cis_graph_builder.py` - IEEE-CIS specific processing

---

## Version History

### v1.0 (2024-12-23)
- Initial release
- Complete ablation framework
- Documentation suite
- Automation scripts
- Test utilities

---

## Support & Contact

### For Questions
1. Check the appropriate documentation file (see navigation table above)
2. Review code comments in `ablation_study.py`
3. Run `test_ablation_setup.py` to diagnose issues

### For Bug Reports
Include:
- Error message
- Output of `test_ablation_setup.py`
- Python version and package versions
- Dataset being used

### For Feature Requests
Consider:
- Is it generalizable to other ablation studies?
- Can it be added without breaking existing functionality?
- Is there a workaround using Python API?

---

**Last Updated**: 2024-12-23
**Package Version**: 1.0
**Maintainer**: FSD-GNN Team
