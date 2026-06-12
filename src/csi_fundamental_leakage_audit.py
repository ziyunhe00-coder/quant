from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from csi_fundamental_improvement_analysis import columns_for_spec, model_specs
from csi_fundamental_rf_small import VALUATION_RENAME
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_adaptive_walkforward import make_folds
from random_forest_a_share_strategy import clean_for_json


BANNED_FEATURE_PATTERNS = (
    "fwd_",
    "target",
    "future",
    "next_",
    "exit_",
    "exec_",
)


def audit_feature_columns(features: pd.DataFrame) -> list[dict]:
    rows = []
    for spec in model_specs(features):
        cols = columns_for_spec(features, spec)
        banned = [col for col in cols if any(pattern in col.lower() for pattern in BANNED_FEATURE_PATTERNS)]
        date_like = [col for col in cols if re.search(r"(notice|report|effective|date|year)", col, re.I)]
        rows.append(
            {
                "model_id": spec.model_id,
                "feature_set": spec.feature_set,
                "feature_count": len(cols),
                "banned_feature_count": len(banned),
                "banned_features": ",".join(banned),
                "date_like_feature_count": len(date_like),
                "date_like_features": ",".join(date_like),
            }
        )
    return rows


def audit_financial_timing(features: pd.DataFrame) -> dict:
    out = {}
    dated = features.dropna(subset=["notice_date", "effective_date"]).copy()
    out["rows_with_notice"] = int(len(dated))
    if len(dated):
        out["notice_date_ge_trade_date"] = int((dated["notice_date"] >= dated["date"]).sum())
        out["effective_date_gt_trade_date"] = int((dated["effective_date"] > dated["date"]).sum())
        out["report_date_gt_notice_date"] = int((dated["report_date"] > dated["notice_date"]).sum())
        out["min_days_since_notice"] = float((dated["date"] - dated["notice_date"]).dt.days.min())
        out["min_days_since_effective"] = float((dated["date"] - dated["effective_date"]).dt.days.min())
    fin_cols = [col for col in features.columns if col.startswith("fin_") and col != "fin_days_since_notice"]
    if fin_cols:
        any_fin = features[fin_cols].notna().any(axis=1)
        out["financial_values_without_effective_date"] = int((any_fin & features["effective_date"].isna()).sum())
    return out


def audit_fold_embargo(features: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    rows = []
    for fold_id, (train_dates, val_dates, test_dates) in enumerate(
        make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo),
        start=1,
    ):
        train_val_gap = int(np.where(dates == val_dates[0])[0][0] - np.where(dates == train_dates[-1])[0][0] - 1)
        val_test_gap = int(np.where(dates == test_dates[0])[0][0] - np.where(dates == val_dates[-1])[0][0] - 1)
        rows.append(
            {
                "fold_id": fold_id,
                "train_end": str(train_dates[-1].date()),
                "val_start": str(val_dates[0].date()),
                "val_end": str(val_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "train_val_gap_trading_days": train_val_gap,
                "val_test_gap_trading_days": val_test_gap,
                "passes_embargo": bool(train_val_gap >= args.embargo and val_test_gap >= args.embargo),
            }
        )
    return rows


def audit_valuation_shift(features: pd.DataFrame, sample_symbols: int) -> dict:
    raw_dir = Path("data/raw_fundamentals_em/valuation")
    checks = []
    symbols = sorted(features["symbol"].dropna().astype(str).str.zfill(6).unique())[:sample_symbols]
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        if "数据日期" not in raw.columns or "总市值" not in raw.columns:
            continue
        raw["date"] = pd.to_datetime(raw["数据日期"], errors="coerce")
        raw["raw_val_total_mv"] = pd.to_numeric(raw["总市值"], errors="coerce")
        raw = raw[["date", "raw_val_total_mv"]].dropna().sort_values("date")
        shifted = raw.copy()
        shifted["expected_val_total_mv"] = shifted["raw_val_total_mv"].shift(1)
        sample = features[features["symbol"].astype(str).str.zfill(6) == symbol][["date", "val_total_mv"]].dropna()
        merged = sample.merge(shifted[["date", "expected_val_total_mv"]], on="date", how="inner").dropna()
        if merged.empty:
            continue
        diff = (merged["val_total_mv"] - merged["expected_val_total_mv"]).abs()
        denom = merged["expected_val_total_mv"].abs().replace(0, np.nan)
        rel = (diff / denom).replace([np.inf, -np.inf], np.nan).dropna()
        checks.append(
            {
                "symbol": symbol,
                "matched_rows": int(len(merged)),
                "max_relative_diff": float(rel.max()) if len(rel) else None,
                "pass_shift_check": bool(len(rel) and rel.max() < 1e-9),
            }
        )
    return {
        "valuation_raw_files_checked": len(checks),
        "valuation_shift_passed": int(sum(item["pass_shift_check"] for item in checks)),
        "checks": checks,
    }


def run(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    features = load_cached_features(args.max_symbols)
    feature_rows = audit_feature_columns(features)
    financial = audit_financial_timing(features)
    embargo = audit_fold_embargo(features, args)
    valuation = audit_valuation_shift(features, args.sample_symbols)
    universe_warning = {
        "status": "warning",
        "issue": "Universe selection in csi_fundamental_rf_small.select_liquid_symbols uses the latest one-year liquidity as of the dataset end date.",
        "impact": "This is a survivorship / future universe selection bias. It does not leak per-row features, but it can overstate historical Sharpe by excluding names that would not have been known as future-liquid survivors.",
        "recommended_fix": "Use point-in-time index constituents or date-by-date rolling liquidity eligibility before final deployment.",
    }
    qfq_warning = {
        "status": "warning",
        "issue": "Historical prices are qfq-adjusted. Return features are mostly ratio-based, but lot-size and actual executable price checks are only approximate on old dates.",
        "recommended_fix": "Use unadjusted OHLC for execution and adjusted returns for research, or maintain split/dividend adjustment factors separately.",
    }
    result = {
        "objective": "Leakage and implementation-bias audit for CSI1000 fundamental RF strategy.",
        "feature_column_audit": feature_rows,
        "financial_timing_audit": financial,
        "fold_embargo_audit": embargo,
        "valuation_shift_audit": valuation,
        "known_biases": [universe_warning, qfq_warning],
        "overall": {
            "hard_future_feature_leak_found": any(row["banned_feature_count"] > 0 for row in feature_rows)
            or financial.get("notice_date_ge_trade_date", 0) > 0
            or financial.get("effective_date_gt_trade_date", 0) > 0
            or not all(row["passes_embargo"] for row in embargo),
            "universe_selection_bias_found": True,
            "qfq_execution_approximation_found": True,
        },
    }
    pd.DataFrame(feature_rows).to_csv(report_dir / "feature_column_audit.csv", index=False)
    pd.DataFrame(embargo).to_csv(report_dir / "fold_embargo_audit.csv", index=False)
    pd.DataFrame(valuation["checks"]).to_csv(report_dir / "valuation_shift_audit.csv", index=False)
    (report_dir / "leakage_audit_summary.json").write_text(json.dumps(clean_for_json(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_for_json(result), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=200)
    parser.add_argument("--report-dir", default="reports/csi_fundamental_rf_small/leakage_audit_n200_v1")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=194)
    parser.add_argument("--test-len", type=int, default=194)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--sample-symbols", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
