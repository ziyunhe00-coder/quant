from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from csi_medium_term_strategy import (
    REPORT_BASE,
    MediumRule,
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


REPORT_DIR = REPORT_BASE / "adaptive_walkforward"


def candidate_rules() -> list[MediumRule]:
    base = rules()
    extra = [
        replace(
            base[6],
            rule_id=101,
            change="Top8相对强度，放宽为12/16止损。",
            hard_stop=-0.12,
            trailing_stop=-0.16,
            max_holding_days=60,
        ),
        replace(
            base[6],
            rule_id=102,
            change="Top12相对强度，放宽为12/18止损。",
            top_n=12,
            max_weight=0.12,
            hard_stop=-0.12,
            trailing_stop=-0.18,
            max_holding_days=70,
        ),
        replace(
            base[4],
            rule_id=103,
            change="Top20相对强度，20日调仓，12/18止损。",
            top_n=20,
            max_weight=0.08,
            hard_stop=-0.12,
            trailing_stop=-0.18,
            max_holding_days=80,
        ),
        replace(
            base[1],
            rule_id=104,
            change="Top15质量趋势，20日调仓，12/18止损。",
            top_n=15,
            max_weight=0.10,
            hard_stop=-0.12,
            trailing_stop=-0.18,
            max_holding_days=80,
        ),
        replace(
            base[5],
            rule_id=105,
            change="Top30低波质量，放宽10/16止损。",
            hard_stop=-0.10,
            trailing_stop=-0.16,
            max_holding_days=90,
        ),
        replace(
            base[6],
            rule_id=106,
            change="Top12相对强度，趋势市场即可交易，12/18止损。",
            top_n=12,
            market_filter="trend_ok",
            max_weight=0.12,
            hard_stop=-0.12,
            trailing_stop=-0.18,
            max_holding_days=70,
        ),
        replace(
            base[4],
            rule_id=107,
            change="Top20相对强度，防守市场过滤，12/18止损。",
            top_n=20,
            market_filter="defensive",
            max_weight=0.08,
            hard_stop=-0.12,
            trailing_stop=-0.18,
            max_holding_days=80,
        ),
        replace(
            base[1],
            rule_id=108,
            change="Top20质量趋势，防守市场过滤，12/18止损。",
            top_n=20,
            market_filter="defensive",
            max_weight=0.08,
            hard_stop=-0.12,
            trailing_stop=-0.18,
            max_holding_days=90,
        ),
    ]
    return base + extra


def make_folds(dates: pd.Index, train_min: int, val_len: int, test_len: int, embargo: int):
    folds = []
    test_start = train_min + embargo + val_len + embargo
    while test_start + test_len <= len(dates):
        val_end = test_start - embargo
        val_start = val_end - val_len
        train_end = val_start - embargo
        if train_end >= train_min:
            folds.append((dates[:train_end], dates[val_start:val_end], dates[test_start : test_start + test_len]))
        test_start += test_len
    return folds


def score_validation(m: dict[str, float]) -> float:
    exposure_penalty = max(0.0, 0.35 - m.get("avg_exposure", 0.0)) * 3.0
    position_penalty = max(0.0, 3.0 - m.get("avg_positions", 0.0)) * 0.15
    dd_penalty = max(0.0, abs(m.get("max_drawdown", 0.0)) - 0.18) * 1.2
    turnover_penalty = max(0.0, m.get("avg_turnover", 0.0) - 0.20) * 0.5
    return m["sharpe"] + 0.25 * m.get("win_rate", 0.0) - exposure_penalty - position_penalty - dd_penalty - turnover_penalty


def sharpe_from_returns(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252))


def block_bootstrap(returns: pd.Series, n: int, block_len: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = returns.fillna(0).to_numpy()
    starts = np.arange(0, len(values) - block_len + 1)
    if len(starts) == 0:
        return {}
    sharpes = []
    for _ in range(n):
        sample = []
        while len(sample) < len(values):
            s = int(rng.choice(starts))
            sample.extend(values[s : s + block_len])
        sharpes.append(sharpe_from_returns(pd.Series(sample[: len(values)])))
    q = np.quantile(sharpes, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_DIR / args.report_name / args.market
    report_dir.mkdir(parents=True, exist_ok=True)
    features = clean_anomalies(load_features(args.market))
    dates = pd.Index(sorted(features.dropna(subset=["ret_60", "fwd_return_40"])["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    model_list = models()
    rule_list = candidate_rules()
    fold_rows = []
    equity_frames = []
    selection_rows = []

    for fold_id, (train_dates, val_dates, test_dates) in enumerate(folds, start=1):
        print(
            f"[fold {fold_id}] train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"val {val_dates[0].date()}..{val_dates[-1].date()} "
            f"test {test_dates[0].date()}..{test_dates[-1].date()}",
            flush=True,
        )
        train = features[features["date"].isin(train_dates)].copy()
        val = features[features["date"].isin(val_dates)].copy()
        test = features[features["date"].isin(test_dates)].copy()
        best = None
        for model_config in model_list:
            model = fit_model(train, model_config)
            val_pred = predict(model, val)
            for rule in rule_list:
                val_bt, _ = backtest(val_pred, rule)
                val_m = metrics(val_bt)
                row = {
                    "fold_id": fold_id,
                    "model_id": model_config.model_id,
                    "rule_id": rule.rule_id,
                    "score": score_validation(val_m),
                    **{f"validation_{k}": v for k, v in val_m.items()},
                }
                selection_rows.append(row)
                if best is None or row["score"] > best["score"]:
                    best = {**row, "model": model_config, "rule": rule}

        selected_model = best["model"]
        selected_rule = best["rule"]
        # The validation block is still in the past and separated from test by embargo, so it can be included.
        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        final_model = fit_model(train_full, selected_model)
        test_pred = predict(final_model, test)
        test_bt, test_holdings = backtest(test_pred, selected_rule)
        test_bt["fold_id"] = fold_id
        equity_frames.append(test_bt)
        test_holdings.to_csv(report_dir / f"fold_{fold_id}_holdings.csv", index=False)
        test_m = metrics(test_bt)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "selected_model_id": selected_model.model_id,
                "selected_rule_id": selected_rule.rule_id,
                "selected_model_change": selected_model.change,
                "selected_rule_change": selected_rule.change,
                "validation_score": best["score"],
                "validation_sharpe": best["validation_sharpe"],
                **{f"test_{k}": v for k, v in test_m.items()},
            }
        )

    selections = pd.DataFrame(selection_rows)
    fold_results = pd.DataFrame(fold_rows)
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    if not equity.empty:
        equity["combined_equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    selections.to_csv(report_dir / "selection_candidates_by_fold.csv", index=False)
    fold_results.to_csv(report_dir / "adaptive_walkforward_folds.csv", index=False)
    equity.to_csv(report_dir / "adaptive_walkforward_equity.csv", index=False)

    if not equity.empty:
        combined_bt = equity.copy()
        combined_bt["equity"] = (1 + combined_bt["net_return"].fillna(0)).cumprod()
        combined = metrics(combined_bt)
    else:
        combined = {}
    returns = equity["net_return"].fillna(0) if not equity.empty else pd.Series(dtype=float)
    summary = {
        "objective": "Adaptive RF walk-forward: select model/rule only on past validation, then test future window.",
        "market": args.market,
        "dataset": {
            "symbols": int(features["symbol"].nunique()),
            "rows": int(len(features)),
            "start": str(features["date"].min().date()),
            "end": str(features["date"].max().date()),
        },
        "fold_settings": {
            "folds": len(folds),
            "train_min": args.train_min,
            "val_len": args.val_len,
            "test_len": args.test_len,
            "embargo": args.embargo,
        },
        "combined": clean_for_json(combined),
        "folds": {
            "avg_test_sharpe": float(fold_results["test_sharpe"].mean()) if len(fold_results) else 0.0,
            "median_test_sharpe": float(fold_results["test_sharpe"].median()) if len(fold_results) else 0.0,
            "min_test_sharpe": float(fold_results["test_sharpe"].min()) if len(fold_results) else 0.0,
            "positive_folds": int((fold_results["test_sharpe"] > 0).sum()) if len(fold_results) else 0,
        },
        "bootstrap_sharpe": block_bootstrap(returns, args.bootstrap, args.block_len, args.seed),
        "selected_rules": clean_for_json(fold_results["selected_rule_id"].value_counts().to_dict()) if len(fold_results) else {},
    }
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["csi1000", "csi500", "combined"], default="csi1000")
    parser.add_argument("--report-name", default="v2")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=126)
    parser.add_argument("--test-len", type=int, default=126)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
