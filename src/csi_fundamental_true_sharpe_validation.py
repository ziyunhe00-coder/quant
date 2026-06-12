from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csi_fundamental_improvement_analysis import (
    attach_industry,
    columns_for_spec,
    diversified_rules,
    fit_model,
    model_specs,
    predict_frame,
)
from csi_fundamental_regime_exposure import ExposureRule, combined_metrics, dynamic_backtest, regime_score
from csi_fundamental_rf_small import REPORT_BASE
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_adaptive_walkforward import block_bootstrap, make_folds
from csi_medium_term_strategy import metrics
from random_forest_a_share_strategy import clean_for_json


def exposure_candidates(vol_budgets: list[float]) -> list[ExposureRule]:
    rules = [ExposureRule(2, "trend_breadth_keep", "Original trend/breadth exposure guard.", "trend_breadth")]
    for vol in vol_budgets:
        code = int(round(vol * 1000))
        rules.extend(
            [
                ExposureRule(
                    code,
                    f"vol_{vol:.2f}",
                    f"Volatility budget {vol:.0%}.",
                    "vol_target",
                    min_exposure=0.15,
                    max_exposure=0.85,
                    target_vol=vol,
                ),
                ExposureRule(
                    code + 1000,
                    f"trend_vol_{vol:.2f}",
                    f"Trend/breadth plus volatility budget {vol:.0%}.",
                    "trend_vol",
                    min_exposure=0.15,
                    max_exposure=0.85,
                    target_vol=vol,
                ),
                ExposureRule(
                    code + 2000,
                    f"trend_vol_dd_{vol:.2f}",
                    f"Trend/breadth plus volatility budget {vol:.0%} and drawdown cut.",
                    "trend_vol_dd",
                    min_exposure=0.15,
                    max_exposure=0.85,
                    target_vol=vol,
                ),
            ]
        )
    return rules


def evaluate_stress(equity: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in [14, 30, 50]:
        tmp = equity.copy()
        tmp["net_return"] = tmp["gross_return"] - tmp["turnover"] * bps / 10000
        rows.append({"case": f"cost_{bps}bps", "dropped_fold": None, **combined_metrics(tmp)})
    best_fold = int(folds.sort_values("test_sharpe", ascending=False).iloc[0]["fold_id"])
    worst_fold = int(folds.sort_values("test_sharpe").iloc[0]["fold_id"])
    last_fold = int(folds["fold_id"].max())
    for label, fold_id in [
        ("drop_best_fold", best_fold),
        ("drop_worst_fold", worst_fold),
        ("drop_last_fold", last_fold),
    ]:
        rows.append({"case": label, "dropped_fold": fold_id, **combined_metrics(equity[equity["fold_id"] != fold_id])})
    return pd.DataFrame(rows)


def run_random_baseline(
    features: pd.DataFrame,
    folds,
    selected_folds: pd.DataFrame,
    candidates: dict[str, ExposureRule],
    simulations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    trade_rules = {rule.rule_id: rule for rule in diversified_rules(constrained_only=True)}
    rows = []
    for sim in range(simulations):
        frames = []
        for fold_id, (_, _, test_dates) in enumerate(folds, start=1):
            selected = selected_folds[selected_folds["fold_id"] == fold_id].iloc[0]
            test = features[features["date"].isin(test_dates)].copy()
            test["prediction"] = rng.normal(size=len(test))
            bt, _ = dynamic_backtest(
                test,
                trade_rules[int(selected["selected_trade_rule_id"])],
                candidates[str(selected["selected_exposure_name"])],
            )
            bt["fold_id"] = fold_id
            frames.append(bt)
        equity = pd.concat(frames, ignore_index=True)
        rows.append({"sim": sim, **combined_metrics(equity)})
    sims = pd.DataFrame(rows)
    return sims, {
        "simulations": int(len(sims)),
        "random_median_sharpe": float(sims["sharpe"].median()),
        "random_p95_sharpe": float(sims["sharpe"].quantile(0.95)),
        "random_max_sharpe": float(sims["sharpe"].max()),
    }


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"true_sharpe_n{args.max_symbols}_{args.report_name}"
    report_dir.mkdir(parents=True, exist_ok=True)

    features = load_cached_features(args.max_symbols)
    features = attach_industry(features, Path(args.industry_map), args.industry_col)
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    vol_budgets = [float(item) for item in args.vol_budgets.split(",")]
    exp_rules = exposure_candidates(vol_budgets)

    fold_rows = []
    selection_rows = []
    equity_frames = []
    holding_frames = []

    for fold_id, (train_dates, val_dates, test_dates) in enumerate(folds, start=1):
        print(
            f"[fold {fold_id}/{len(folds)}] train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"val {val_dates[0].date()}..{val_dates[-1].date()} test {test_dates[0].date()}..{test_dates[-1].date()}",
            flush=True,
        )
        train = features[features["date"].isin(train_dates)].copy()
        val = features[features["date"].isin(val_dates)].copy()
        test = features[features["date"].isin(test_dates)].copy()
        best = None
        rows = []
        for spec in model_specs(features):
            cols = columns_for_spec(features, spec)
            model = fit_model(train, spec, cols)
            val_pred = predict_frame(model, val, cols)
            for trade_rule in diversified_rules(constrained_only=True):
                for exp_rule in exp_rules:
                    val_bt, _ = dynamic_backtest(val_pred, trade_rule, exp_rule)
                    val_m = metrics(val_bt)
                    score = regime_score(val_m, args.score_mode)
                    row = {
                        "fold_id": fold_id,
                        "model_id": spec.model_id,
                        "feature_set": spec.feature_set,
                        "trade_rule_id": trade_rule.rule_id,
                        "exposure_name": exp_rule.name,
                        "target_vol": exp_rule.target_vol,
                        "score": score,
                        **{f"validation_{key}": value for key, value in val_m.items()},
                    }
                    rows.append(row)
                    if best is None or score > best["score"]:
                        best = {
                            **row,
                            "model": spec,
                            "cols": cols,
                            "trade_rule": trade_rule,
                            "exposure_rule": exp_rule,
                        }
        selection_rows.extend(rows)

        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        model = fit_model(train_full, best["model"], best["cols"])
        pred = predict_frame(model, test, best["cols"])
        bt, holdings = dynamic_backtest(pred, best["trade_rule"], best["exposure_rule"])
        bt["fold_id"] = fold_id
        holdings["fold_id"] = fold_id
        equity_frames.append(bt)
        holding_frames.append(holdings)
        test_m = metrics(bt)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "selected_model_id": best["model"].model_id,
                "selected_feature_set": best["model"].feature_set,
                "selected_trade_rule_id": best["trade_rule"].rule_id,
                "selected_exposure_name": best["exposure_rule"].name,
                "selected_target_vol": best["exposure_rule"].target_vol,
                "validation_score": best["score"],
                "validation_sharpe": best["validation_sharpe"],
                "validation_annual_return": best["validation_annual_return"],
                "validation_avg_exposure": best["validation_avg_exposure"],
                **{f"test_{key}": value for key, value in test_m.items()},
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)
    equity = pd.concat(equity_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)
    equity["combined_equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    combined = combined_metrics(equity)
    returns = equity["net_return"].fillna(0)
    stress = evaluate_stress(equity, folds_df)

    candidate_map = {rule.name: rule for rule in exp_rules}
    sims, random_summary = run_random_baseline(features, folds, folds_df, candidate_map, args.random_sims, args.seed + 202)
    random_summary["observed_sharpe"] = combined["sharpe"]
    random_summary["p_value_random_ge_observed"] = float((sims["sharpe"] >= combined["sharpe"]).mean())

    summary = {
        "objective": "Nested true-Sharpe check: model, trade rule, and exposure budget selected only on validation folds.",
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
            "score_mode": args.score_mode,
            "vol_budgets": vol_budgets,
        },
        "combined": clean_for_json(combined),
        "folds": {
            "avg_test_sharpe": float(folds_df["test_sharpe"].mean()),
            "median_test_sharpe": float(folds_df["test_sharpe"].median()),
            "min_test_sharpe": float(folds_df["test_sharpe"].min()),
            "max_test_sharpe": float(folds_df["test_sharpe"].max()),
            "positive_folds": int((folds_df["test_sharpe"] > 0).sum()),
        },
        "bootstrap_sharpe": block_bootstrap(returns, args.bootstrap, args.block_len, args.seed),
        "selected_exposure_rules": clean_for_json(folds_df["selected_exposure_name"].value_counts().to_dict()),
        "selected_target_vols": clean_for_json(folds_df["selected_target_vol"].value_counts().to_dict()),
        "random_baseline": clean_for_json(random_summary),
    }

    folds_df.to_csv(report_dir / "nested_folds.csv", index=False)
    selections.to_csv(report_dir / "nested_validation_candidates.csv", index=False)
    equity.to_csv(report_dir / "nested_equity.csv", index=False)
    holdings.to_csv(report_dir / "nested_holdings.csv", index=False)
    stress.to_csv(report_dir / "nested_stress_checks.csv", index=False)
    sims.to_csv(report_dir / "nested_random_signal_baseline.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nNested folds:")
    print(
        folds_df[
            [
                "fold_id",
                "selected_feature_set",
                "selected_trade_rule_id",
                "selected_exposure_name",
                "selected_target_vol",
                "validation_sharpe",
                "validation_annual_return",
                "test_sharpe",
                "test_annual_return",
                "test_max_drawdown",
                "test_avg_exposure",
            ]
        ].to_string(index=False)
    )
    print("\nSummary:")
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=200)
    parser.add_argument("--report-name", default="v1")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=194)
    parser.add_argument("--test-len", type=int, default=194)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-sims", type=int, default=200)
    parser.add_argument("--score-mode", choices=["sharpe", "return_balance"], default="return_balance")
    parser.add_argument("--vol-budgets", default="0.18,0.20,0.22,0.24,0.26,0.28,0.30")
    parser.add_argument("--industry-map", default="data/processed/csi1000_industry_map_200.csv")
    parser.add_argument("--industry-col", default="industry_gate")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
