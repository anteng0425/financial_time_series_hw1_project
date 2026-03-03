# Financial Time Series Analysis — Homework 1

This project reproduces the numerical results (test statistics, p-values, and confidence intervals) used in the Homework 1 write-up.

## Contents
- `data/`
  - `Data.xlsx`
  - `TSMC.xlsx`
  - (optional) `E371HW1.pdf`
- `src/run_analysis.py` — main script to recompute all answers
- `outputs/` — results will be written here (`results.json`, `results.md`)

## Requirements
- Python 3.10+ recommended

It is recommended to create and activate a virtual environment before installing dependencies and running the project.

Create a virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:

On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies inside the virtual environment:
```bash
pip install -r requirements.txt
```

## Run
From the project root:
```bash
python src/run_analysis.py
```

## Output
After running, check:
- `outputs/results.md` (human-readable summary)
- `outputs/results.json` (machine-readable)

## Notes on definitions
- Sample variance uses the unbiased estimator: denominator `n-1`.
- Mean tests use the one-sample t statistic: `(xbar - mu0) / (s / sqrt(n))`, two-sided.
- Variance tests use the chi-square statistic: `(n-1)*s^2 / sigma0^2`, two-sided (via both tails).
- Log return is computed as: `r_t = ln(P_t) - ln(P_{t-1})`.
