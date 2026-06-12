from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csi_fundamental_improvement_analysis import attach_industry, diversified_backtest, diversified_rules
from csi_fundamental_rf_small import REPORT_BASE
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_adaptive_walkforward import make_folds
from csi_medium_term_strategy import metrics
from random_forest_a_share_strategy import clean_for_json


def run_random_once(
    features: pd.DataFrame,
    folds: list[tuple[pd.Index, pd.Index, pd.Index]],
    selected_rules: dict[int, int],
    rng: np.random.Generator,
    constrained_only: bool,
) -> pd.DataFrame:
    rule_by_id = {rule.rule_id: rule for rule in diversified_rules(constrained_only)}
    equity_frames = []
    for fold_id, (_, _, test_dates) in enumerate(folds, start=1):
        test = features[features["date"].isin(test_dates)].copy()
        test["prediction"] = rng.random(len(test))
        rule = rule_by_id[int(selected_rules[fold_id])]
        bt, _ = diversified_backtest(test, rule)
        bt["fold_id"] = fold_id
        equity_frames.append(bt)
    equity = pd.concat(equity_frames, ignore_index=True)
    equity["equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    return equity


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"improvement_n{args.max_symbols}_{args.report_name}"
    if not report_dir.exists():
        raise FileNotFoundError(report_dir)
    folds_df = pd.read_csv(report_dir / "improved_walkforward_folds.csv")
    observed_eq = pd.read_csv(report_dir / "improved_walkforward_equity.csv")
    observed_eq["equity"] = (1 + observed_eq["net_return"].fillna(0)).cumprod()
    observed = metrics(observed_eq)

    features = load_cached_features(args.max_symbols)
    features = attach_industry(features, Path(args.industry_map), args.industry_col)
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    if len(folds) != len(folds_df):
        raise RuntimeError(f"Fold count mismatch: generated {len(folds)}, report has {len(folds_df)}")
    selected_rules = dict(zip(folds_df["fold_id"].astype(int), folds_df["selected_rule_id"].astype(int)))

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.simulations):
        sim_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        equity = run_random_once(features, folds, selected_rules, sim_rng, args.constrained_only)
        m = metrics(equity)
        rows.append({"simulation": i + 1, **m})
        if (i + 1) % 50 == 0:
            print(f"[random] {i + 1}/{args.simulations}", flush=True)
    random_metrics = pd.DataFrame(rows)
    random_metrics.to_csv(report_dir / "random_signal_baseline.csv", index=False)
    p_value = float(((random_metrics["sharpe"] >= observed["sharpe"]).sum() + 1) / (len(random_metrics) + 1))
    q = random_metrics["sharpe"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    summary = {
        "objective": "Random prediction baseline using the same selected rules, market filters, industry caps, and exposure caps.",
        "report_name": args.report_name,
        "simulations": args.simulations,
        "observed": clean_for_json(observed),
        "random_sharpe_quantiles": {
            "p05": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "median": float(q.loc[0.5]),
            "p75": float(q.loc[0.75]),
            "p95": float(q.loc[0.95]),
        },
        "random_mean_sharpe": float(random_metrics["sharpe"].mean()),
        "random_std_sharpe": float(random_metrics["sharpe"].std(ddof=0)),
        "p_value_random_ge_observed": p_value,
    }
    (report_dir / "random_signal_baseline_summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=120)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=194)
    parser.add_argument("--test-len", type=int, default=194)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--industry-map", default="data/processed/csi1000_industry_map_120.csv")
    parser.add_argument("--industry-col", default="industry_gate")
    parser.add_argument("--constrained-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
