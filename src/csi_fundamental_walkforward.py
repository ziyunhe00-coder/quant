from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from csi_fundamental_rf_small import (
    REPORT_BASE,
    candidate_rules,
    feature_columns,
    fit_rf,
    model_configs,
    predict_rf,
    score_validation,
)
from csi_medium_term_adaptive_walkforward import block_bootstrap, make_folds
from csi_medium_term_strategy import backtest, metrics
from random_forest_a_share_strategy import PROCESSED_DIR, clean_for_json


def load_cached_features(max_symbols: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"csi1000_fundamental_small_{max_symbols}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached feature file: {path}")
    features = pd.read_parquet(path)
    features["date"] = pd.to_datetime(features["date"])
    features["symbol"] = features["symbol"].astype(str).str.zfill(6)
    features = features[features["amount"] > 0].copy()
    return features.replace([np.inf, -np.inf], np.nan).sort_values(["date", "symbol"])


def combined_metrics(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    out = equity.copy()
    out["equity"] = (1 + out["net_return"].fillna(0)).cumprod()
    return metrics(out)


def select_on_validation(train: pd.DataFrame, val: pd.DataFrame, features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    for config in model_configs():
        cols = feature_columns(features, config)
        model = fit_rf(train, config, cols)
        val_pred = predict_rf(model, val, cols)
        for rule in candidate_rules():
            val_bt, _ = backtest(val_pred, rule)
            val_m = metrics(val_bt)
            row = {
                "model_id": config.model_id,
                "rule_id": rule.rule_id,
                "feature_set": config.feature_set,
                "feature_count": len(cols),
                "score": score_validation(val_m),
                "model_change": config.change,
                "rule_change": rule.change,
                **{f"validation_{key}": value for key, value in val_m.items()},
            }
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = {**row, "model": config, "rule": rule, "cols": cols}
    return best, pd.DataFrame(rows)


def run_test_window(
    train_full: pd.DataFrame,
    test: pd.DataFrame,
    features: pd.DataFrame,
    model_id: int,
    rule_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, object, object, list[str]]:
    config = next(item for item in model_configs() if item.model_id == model_id)
    rule = next(item for item in candidate_rules() if item.rule_id == rule_id)
    cols = feature_columns(features, config)
    model = fit_rf(train_full, config, cols)
    pred = predict_rf(model, test, cols)
    bt, holdings = backtest(pred, rule)
    return bt, holdings, config, rule, cols


def summarize_folds(fold_results: pd.DataFrame, equity: pd.DataFrame, args: argparse.Namespace) -> dict:
    combined = combined_metrics(equity)
    returns = equity["net_return"].fillna(0) if not equity.empty else pd.Series(dtype=float)
    return {
        "combined": clean_for_json(combined),
        "folds": {
            "count": int(len(fold_results)),
            "avg_test_sharpe": float(fold_results["test_sharpe"].mean()) if len(fold_results) else 0.0,
            "median_test_sharpe": float(fold_results["test_sharpe"].median()) if len(fold_results) else 0.0,
            "min_test_sharpe": float(fold_results["test_sharpe"].min()) if len(fold_results) else 0.0,
            "max_test_sharpe": float(fold_results["test_sharpe"].max()) if len(fold_results) else 0.0,
            "positive_folds": int((fold_results["test_sharpe"] > 0).sum()) if len(fold_results) else 0,
        },
        "bootstrap_sharpe": block_bootstrap(returns, args.bootstrap, args.block_len, args.seed),
    }


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"walkforward_n{args.max_symbols}_{args.report_name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    features = load_cached_features(args.max_symbols)
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    if not folds:
        raise RuntimeError("No folds generated. Lower train_min/val_len/test_len or check data length.")

    adaptive_fold_rows = []
    adaptive_selection_rows = []
    adaptive_equity_frames = []
    locked_fold_rows = []
    locked_equity_frames = []

    for fold_id, (train_dates, val_dates, test_dates) in enumerate(folds, start=1):
        print(
            f"[fold {fold_id}/{len(folds)}] "
            f"train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"val {val_dates[0].date()}..{val_dates[-1].date()} "
            f"test {test_dates[0].date()}..{test_dates[-1].date()}",
            flush=True,
        )
        train = features[features["date"].isin(train_dates)].copy()
        val = features[features["date"].isin(val_dates)].copy()
        test = features[features["date"].isin(test_dates)].copy()

        best, selection = select_on_validation(train, val, features)
        selection["fold_id"] = fold_id
        adaptive_selection_rows.append(selection)

        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        adaptive_bt, adaptive_holdings, selected_model, selected_rule, selected_cols = run_test_window(
            train_full,
            test,
            features,
            int(best["model_id"]),
            int(best["rule_id"]),
        )
        adaptive_bt["fold_id"] = fold_id
        adaptive_equity_frames.append(adaptive_bt)
        adaptive_holdings.to_csv(report_dir / f"adaptive_fold_{fold_id}_holdings.csv", index=False)
        adaptive_m = metrics(adaptive_bt)
        adaptive_fold_rows.append(
            {
                "fold_id": fold_id,
                "selected_model_id": selected_model.model_id,
                "selected_rule_id": selected_rule.rule_id,
                "selected_feature_set": selected_model.feature_set,
                "selected_feature_count": len(selected_cols),
                "selected_model_change": selected_model.change,
                "selected_rule_change": selected_rule.change,
                "validation_score": best["score"],
                "validation_sharpe": best["validation_sharpe"],
                **{f"test_{key}": value for key, value in adaptive_m.items()},
            }
        )

        locked_bt, locked_holdings, locked_model, locked_rule, locked_cols = run_test_window(
            train_full,
            test,
            features,
            args.locked_model_id,
            args.locked_rule_id,
        )
        locked_bt["fold_id"] = fold_id
        locked_equity_frames.append(locked_bt)
        locked_holdings.to_csv(report_dir / f"locked_fold_{fold_id}_holdings.csv", index=False)
        locked_m = metrics(locked_bt)
        locked_fold_rows.append(
            {
                "fold_id": fold_id,
                "model_id": locked_model.model_id,
                "rule_id": locked_rule.rule_id,
                "feature_set": locked_model.feature_set,
                "feature_count": len(locked_cols),
                **{f"test_{key}": value for key, value in locked_m.items()},
            }
        )

    adaptive_folds = pd.DataFrame(adaptive_fold_rows)
    adaptive_selection = pd.concat(adaptive_selection_rows, ignore_index=True)
    adaptive_equity = pd.concat(adaptive_equity_frames, ignore_index=True)
    locked_folds = pd.DataFrame(locked_fold_rows)
    locked_equity = pd.concat(locked_equity_frames, ignore_index=True)

    adaptive_equity["combined_equity"] = (1 + adaptive_equity["net_return"].fillna(0)).cumprod()
    locked_equity["combined_equity"] = (1 + locked_equity["net_return"].fillna(0)).cumprod()
    adaptive_folds.to_csv(report_dir / "adaptive_walkforward_folds.csv", index=False)
    adaptive_selection.to_csv(report_dir / "adaptive_selection_candidates_by_fold.csv", index=False)
    adaptive_equity.to_csv(report_dir / "adaptive_walkforward_equity.csv", index=False)
    locked_folds.to_csv(report_dir / "locked_walkforward_folds.csv", index=False)
    locked_equity.to_csv(report_dir / "locked_walkforward_equity.csv", index=False)

    locked_model = next(item for item in model_configs() if item.model_id == args.locked_model_id)
    locked_rule = next(item for item in candidate_rules() if item.rule_id == args.locked_rule_id)
    summary = {
        "objective": "Verify whether the fundamental RF Sharpe survives expanding walk-forward validation.",
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
        "adaptive": summarize_folds(adaptive_folds, adaptive_equity, args),
        "adaptive_selected_feature_sets": clean_for_json(adaptive_folds["selected_feature_set"].value_counts().to_dict()),
        "adaptive_selected_rules": clean_for_json(adaptive_folds["selected_rule_id"].value_counts().to_dict()),
        "locked_strategy": {
            "model": asdict(locked_model),
            "rule": asdict(locked_rule),
            **summarize_folds(locked_folds, locked_equity, args),
        },
        "interpretation_note": (
            "Adaptive uses only each fold's past validation to select model/rule. "
            "Locked uses the 80/10/10 best setting and is less pure, but tests parameter robustness."
        ),
    }
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nAdaptive folds:")
    print(
        adaptive_folds[
            [
                "fold_id",
                "selected_feature_set",
                "selected_model_id",
                "selected_rule_id",
                "validation_sharpe",
                "test_sharpe",
                "test_annual_return",
                "test_max_drawdown",
                "test_avg_positions",
            ]
        ].to_string(index=False)
    )
    print("\nLocked folds:")
    print(
        locked_folds[
            [
                "fold_id",
                "feature_set",
                "test_sharpe",
                "test_annual_return",
                "test_max_drawdown",
                "test_avg_positions",
            ]
        ].to_string(index=False)
    )
    print("\nSummary:")
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=120)
    parser.add_argument("--report-name", default="v1")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=126)
    parser.add_argument("--test-len", type=int, default=126)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locked-model-id", type=int, default=4)
    parser.add_argument("--locked-rule-id", type=int, default=201)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
