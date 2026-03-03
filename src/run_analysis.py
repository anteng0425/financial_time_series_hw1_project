#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce computations for Financial Time Series Analysis — Homework 1.

Outputs:
- outputs/results.md
- outputs/results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy import stats


def _pick_first_numeric_column(df: pd.DataFrame, preferred: Optional[List[str]] = None) -> str:
    """
    Pick a numeric column. If `preferred` is given, choose the first match (case-insensitive).
    Otherwise, choose the first column that can be coerced to numeric with at least 80% non-NaN.
    """
    cols = list(df.columns)

    if preferred:
        lower_map = {str(c).lower(): c for c in cols}
        for name in preferred:
            key = str(name).lower()
            if key in lower_map:
                return str(lower_map[key])

    # try numeric coercion
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        ratio = s.notna().mean()
        if ratio >= 0.8:
            return str(c)

    raise ValueError("No suitable numeric column found in the Excel sheet.")


def _two_sided_p_from_tails(cdf: float, sf: float) -> float:
    """Two-sided p-value from cdf and survival function tails: 2*min(cdf, sf), capped at 1."""
    p = 2.0 * min(cdf, sf)
    return float(min(1.0, max(0.0, p)))


def _chi2_two_sided_pvalue(stat: float, df: int) -> Tuple[float, float]:
    """
    Two-sided chi-square p-value computed from both tails.
    Returns (p_value, log_p_value) where log_p_value is useful when p_value underflows to 0.
    """
    # Use logcdf/logsf for stability
    logcdf = float(stats.chi2.logcdf(stat, df))
    logsf = float(stats.chi2.logsf(stat, df))
    min_log_tail = min(logcdf, logsf)
    log_p = math.log(2.0) + min_log_tail

    # Convert to float if possible
    p = float(math.exp(log_p)) if log_p > -745 else 0.0  # exp underflow guard
    # also cap at 1
    if p > 1.0:
        p = 1.0
        log_p = 0.0
    return p, log_p


def _format_p(p: float, log_p: Optional[float] = None) -> str:
    """Format p-value; if extremely small, display as '< 1e-16'."""
    if p > 0:
        # 4 decimal places if not tiny, else scientific
        if p >= 1e-4:
            return f"{p:.4f}"
        return f"{p:.2e}"
    # p == 0, try log_p
    if log_p is not None and log_p < math.log(1e-16):
        return "< 1e-16"
    return "0.0000"


def mean_t_test(x: np.ndarray, mu0: float, alpha: float) -> Dict[str, Any]:
    n = int(x.size)
    xbar = float(np.mean(x))
    s2 = float(np.var(x, ddof=1))
    s = math.sqrt(s2)
    df = n - 1
    t_stat = (xbar - mu0) / (s / math.sqrt(n))
    p = 2.0 * (1.0 - float(stats.t.cdf(abs(t_stat), df=df)))
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci = (xbar - tcrit * s / math.sqrt(n), xbar + tcrit * s / math.sqrt(n))
    return {
        "n": n,
        "xbar": xbar,
        "s2": s2,
        "df": df,
        "t_stat": float(t_stat),
        "p_value": float(p),
        "ci_mean": (float(ci[0]), float(ci[1])),
    }


def var_chi2_test(x: np.ndarray, sigma2_0: float, alpha: float) -> Dict[str, Any]:
    n = int(x.size)
    s2 = float(np.var(x, ddof=1))
    df = n - 1
    chi2_stat = (df * s2) / sigma2_0
    p, log_p = _chi2_two_sided_pvalue(chi2_stat, df=df)

    # CI for variance
    chi2_lo = float(stats.chi2.ppf(1.0 - alpha / 2.0, df=df))
    chi2_hi = float(stats.chi2.ppf(alpha / 2.0, df=df))
    ci = (df * s2 / chi2_lo, df * s2 / chi2_hi)
    return {
        "n": n,
        "s2": s2,
        "df": df,
        "chi2_stat": float(chi2_stat),
        "p_value": float(p),
        "log_p_value": float(log_p),
        "ci_var": (float(ci[0]), float(ci[1])),
    }


def load_series_from_excel(path: Path, sheet: int | str = 0, preferred_cols: Optional[List[str]] = None) -> np.ndarray:
    df = pd.read_excel(path, sheet_name=sheet)
    col = _pick_first_numeric_column(df, preferred=preferred_cols)
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        raise ValueError(f"Loaded series is empty from {path} (column={col}).")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_xlsx", type=str, default="data/Data.xlsx")
    ap.add_argument("--tsmc_xlsx", type=str, default="data/TSMC.xlsx")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_xlsx = (root / args.data_xlsx).resolve()
    tsmc_xlsx = (root / args.tsmc_xlsx).resolve()
    alpha = float(args.alpha)

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Q1: Data.xlsx -----
    x = load_series_from_excel(data_xlsx, preferred_cols=["Observations", "Observation", "X", "x"])
    q1a = mean_t_test(x, mu0=170.0, alpha=alpha)
    q1b = var_chi2_test(x, sigma2_0=9.0, alpha=alpha)

    # ----- Q2: TSMC.xlsx -----
    P = load_series_from_excel(tsmc_xlsx, preferred_cols=["CloseP", "Close", "收盤價", "收盤", "Adj Close", "closep"])
    q2a = mean_t_test(P, mu0=180.0, alpha=alpha)
    q2b = var_chi2_test(P, sigma2_0=3300.0, alpha=alpha)

    # log returns in raw log units
    r = np.log(P[1:]) - np.log(P[:-1])
    q2c = mean_t_test(r, mu0=0.0, alpha=alpha)
    q2d = var_chi2_test(r, sigma2_0=3.0 / (100.0**2), alpha=alpha)

    results: Dict[str, Any] = {
        "alpha": alpha,
        "Q1": {"a_mean_mu0_170": q1a, "b_var_sigma2_0_9": q1b},
        "Q2": {
            "a_close_mean_mu0_180": q2a,
            "b_close_var_sigma2_0_3300": q2b,
            "c_return_mean_mu0_0": q2c,
            "d_return_var_sigma2_0_3": q2d,
        },
        "definitions": {
            "sample_variance": "unbiased, denominator n-1",
            "log_return": "r_t = ln(P_t) - ln(P_{t-1})",
            "mean_test": "two-sided one-sample t test",
            "var_test": "two-sided chi-square test",
        },
    }

    # Write JSON
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write Markdown summary
    md_lines: List[str] = []
    md_lines.append("# Homework 1 — Recomputed Results\n")
    md_lines.append(f"- alpha = {alpha}\n")

    def block(title: str, d: Dict[str, Any], is_chi2: bool = False) -> None:
        md_lines.append(f"## {title}\n")
        md_lines.append(f"- n = {d['n']}\n")
        if "xbar" in d:
            md_lines.append(f"- sample mean = {d['xbar']:.4f}\n")
        if "s2" in d:
            md_lines.append(f"- sample variance = {d['s2']:.4f}\n")
        if "t_stat" in d:
            md_lines.append(f"- t statistic = {d['t_stat']:.4f} (df={d['df']})\n")
            md_lines.append(f"- p-value = {_format_p(d['p_value'])}\n")
            lo, hi = d["ci_mean"]
            md_lines.append(f"- 95% CI for mean = [{lo:.4f}, {hi:.4f}]\n")
        if "chi2_stat" in d:
            p_str = _format_p(d["p_value"], d.get("log_p_value"))
            md_lines.append(f"- chi-square statistic = {d['chi2_stat']:.4f} (df={d['df']})\n")
            md_lines.append(f"- p-value = {p_str}\n")
            lo, hi = d["ci_var"]
            md_lines.append(f"- 95% CI for variance = [{lo:.4f}, {hi:.4f}]\n")

        md_lines.append("\n")

    block("Q1(a) Mean test: mu = 170 (Data.xlsx)", q1a)
    block("Q1(b) Variance test: sigma^2 = 9 (Data.xlsx)", q1b)

    block("Q2(a) Mean test: mu = 180 (TSMC close price)", q2a)
    block("Q2(b) Variance test: sigma^2 = 3300 (TSMC close price)", q2b)
    block("Q2(c) Mean test: mu = 0 (log return)", q2c)
    block("Q2(d) Variance test: sigma^2 = 0.0003 (log return)", q2d)

    (out_dir / "results.md").write_text("".join(md_lines), encoding="utf-8")

    print("Done. Wrote outputs/results.json and outputs/results.md")


if __name__ == "__main__":
    main()
