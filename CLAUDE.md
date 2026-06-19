# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

A teaching collection of ~20 standalone ML example scripts covering regression, classification, clustering, dimensionality reduction, ensemble methods, and neural networks. Each module lives in its own directory with a single `main.py` and is **fully independent** — there are no cross-module imports.

## Running scripts

```bash
python <module_dir>/main.py       # e.g., python 04_knn/main.py
python practice/01_california_housing/main.py
```

All dependencies are in `requirements.txt`. There is no build step, test suite, or package structure.

## Code conventions

**Chinese font support** — every file sets:
```python
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

**Matplotlib backend** — inconsistent across files. Some use `matplotlib.use("Agg")` (headless, saves figures only), others use `matplotlib.use("TkAgg")` (shows `plt.show()` windows). When editing or creating new scripts, prefer `TkAgg` if interactive display is wanted, `Agg` if only saving PNGs. Don't change the backend of existing scripts unless there's a specific reason.

**Data files** — most modules generate synthetic data inline with `np.random.seed(42)` when the expected CSV/txt file is not found. Expected data files live in the top-level `data/` directory. Generation functions typically check `os.path.exists()` before creating.

**Module docstrings** — each `main.py` opens with a docstring listing the sub-tasks it covers (usually 3–4 tasks per module), each illustrating a different variant or application of the core algorithm.

**Output** — all modules print formatted results to stdout (tables, coefficients, metrics) and save figures as PNG files in their own directory.

## Architecture patterns

### Core experiments (`01_*` through `14_*`)
Each follows a progression: data preparation → model construction → training → evaluation → visualization. Many tasks include both a **manual (from-scratch) implementation** and an **sklearn equivalent** for comparison — especially in `04_knn`, `06_naive_bayes`, `07_svm_principles`, `11_dimensionality_reduction`, and `practice/02_german_credit`.

### Practice projects (`practice/0*_*`)
Longer, more applied scripts that combine multiple techniques into a realistic workflow. They tend to have richer data preprocessing (encoding, imputation, feature engineering) and more elaborate visualizations.

### Task numbering convention
Within each `main.py`, tasks are labeled `Task 1`, `Task 2`, `Task 3` (or `任务3.1`, `任务3.2`, etc.) matching the experiment report structure in `实验报告/`. Task 1 is usually a "follow the textbook" warm-up, Task 2 applies the method to a standard dataset, and Task 3 is an independent problem-solving exercise.

## Key constraints

- **No cross-module dependencies**: every script is self-contained. Don't refactor shared utilities into a common module.
- **sklearn is the primary ML library**; `numpy` for manual implementations; `matplotlib` + `seaborn` for visualization.
- **imbalanced-learn** is used for SMOTE in `practice/04_credit_card_fraud`.
- **scipy** is used for statistical tests (Shapiro-Wilk, norm, linregress) in a few modules.
- **pypdf** is in requirements.txt only because it was used during repository creation to extract text from experiment report PDFs; it is not used by any of the ML scripts.
