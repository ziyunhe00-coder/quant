from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from csi_medium_term_strategy import (
    REPORT_BASE,
    backtest,
    clean_anomalies,
    fit_model,
    load_features,
    metrics,
    models,
    predict,
    rules,
)
from random_forest_a_share_strategy import clean_for_json


ROBUST_DIR = REPORT_BASE / "robust"


def fixed_csi1000_strategy():
    model = next(item for item in models() if item.model_id == 2)
    rule = next(item for item in rules() if item.rule_id == 7)
    return model, rule


def make_anchored_folds(dates: pd.Index, train_min: int, test_len: int, embargo: int) -> list[tuple[pd.Index, pd.Index]]:
    folds = []
    start = train_min
    while start + test_len <= len(dates):
        train_end = start - embargo
        if train_end <= 0:
            start += test_len
            continue
        folds.append((dates[:train_end], dates[start : start + test_len]))
        start += test_len
    return folds


def sharpe_from_returns(returns: pd.Series) -> float:
    returns = returns.fillna(0)
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252))


def block_bootstrap_sharpe(returns: pd.Series, n_boot: int, block_len: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = returns.fillna(0).to_numpy()
    if len(values) < block_len:
        return {"p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0}
    sharpes = []
    starts = np.arange(0, len(values) - block_len + 1)
    for _ in range(n_boot):
        sample = []
        while len(sample) < len(values):
            start = int(rng.choice(starts))
            sample.extend(values[start : start + block_len])
        sample = pd.Series(sample[: len(values)])
        sharpes.append(sharpe_from_returns(sample))
    q = np.quantile(sharpes, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {"p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def run_fixed_walkforward(features: pd.DataFrame, cost_bps: float, report_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_config, base_rule = fixed_csi1000_strategy()
    rule = replace(base_rule, transaction_cost_bps=cost_bps)
    dates = pd.Index(sorted(features.dropna(subset=["ret_60", "fwd_return_40"])["date"].unique()))
    folds = make_anchored_folds(dates, args.train_min, args.test_len, args.embargo)
    fold_rows = []
    equity_frames = []
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        print(
            f"[wf cost={cost_bps}] fold {fold_id}: train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"test {test_dates[0].date()}..{test_dates[-1].date()}",
            flush=True,
        )
        train = features[features["date"].isin(train_dates)].copy()
        test = features[features["date"].isin(test_dates)].copy()
        model = fit_model(train, model_config)
        pred = predict(model, test)
        bt, holdings = backtest(pred, rule)
        bt["fold_id"] = fold_id
        bt["cost_bps"] = cost_bps
        equity_frames.append(bt)
        row = {
            "fold_id": fold_id,
            "cost_bps": cost_bps,
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            **metrics(bt),
            "holdings_rows": int(len(holdings)),
        }
        fold_rows.append(row)
    folds_df = pd.DataFrame(fold_rows)
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    folds_df.to_csv(report_dir / f"fixed_walkforward_cost_{int(cost_bps)}_folds.csv", index=False)
    equity.to_csv(report_dir / f"fixed_walkforward_cost_{int(cost_bps)}_equity.csv", index=False)
    return folds_df, equity


def summarize_walkforward(folds: pd.DataFrame, equity: pd.DataFrame, args: argparse.Namespace) -> dict[str, float | int | dict[str, float]]:
    if equity.empty:
        return {}
    combined = metrics(equity.assign(equity=(1 + equity["net_return"]).cumprod()))
    returns = equity["net_return"].fillna(0)
    return {
        "folds": int(len(folds)),
        "combined_sharpe": combined["sharpe"],
        "combined_annual_return": combined["annual_return"],
        "combined_annual_vol": combined["annual_vol"],
        "combined_max_drawdown": combined["max_drawdown"],
        "combined_total_return": combined["total_return"],
        "avg_fold_sharpe": float(folds["sharpe"].mean()),
        "median_fold_sharpe": float(folds["sharpe"].median()),
        "min_fold_sharpe": float(folds["sharpe"].min()),
        "positive_folds": int((folds["sharpe"] > 0).sum()),
        "fold_sharpe_p25": float(folds["sharpe"].quantile(0.25)),
        "fold_sharpe_p75": float(folds["sharpe"].quantile(0.75)),
        "bootstrap_sharpe": block_bootstrap_sharpe(returns, args.bootstrap, args.block_len, args.seed),
    }


def run(args: argparse.Namespace) -> None:
    report_dir = ROBUST_DIR / args.market
    report_dir.mkdir(parents=True, exist_ok=True)
    features = clean_anomalies(load_features(args.market))
    model_config, rule = fixed_csi1000_strategy()

    summaries = {}
    all_folds = []
    for cost in args.cost_bps:
        folds, equity = run_fixed_walkforward(features, cost, report_dir, args)
        summaries[f"cost_{int(cost)}bps"] = summarize_walkforward(folds, equity, args)
        all_folds.append(folds)
    fold_table = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    fold_table.to_csv(report_dir / "all_fixed_walkforward_folds.csv", index=False)

    summary = {
        "objective": "Robust validation for locked CSI1000 medium-term strategy; no final-period parameter selection.",
        "market": args.market,
        "dataset": {
            "symbols": int(features["symbol"].nunique()),
            "rows": int(len(features)),
            "start": str(features["date"].min().date()),
            "end": str(features["date"].max().date()),
        },
        "locked_model": clean_for_json(model_config.__dict__),
        "locked_rule": clean_for_json(rule.__dict__),
        "fold_settings": {
            "train_min_days": args.train_min,
            "test_len_days": args.test_len,
            "embargo_days": args.embargo,
            "bootstrap": args.bootstrap,
            "block_len": args.block_len,
        },
        "summaries": clean_for_json(summaries),
    }
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["csi1000", "csi500", "combined"], default="csi1000")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--test-len", type=int, default=126)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[14.0, 30.0, 50.0])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
