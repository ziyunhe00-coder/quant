from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from csi500_iterative_improvement import (
    REPORT_DIR_CSI,
    RuleConfig,
    backtest,
    final_retrain_check,
    fit_model,
    load_features,
    model_configs,
    predict,
    rule_configs,
)
from adaptive_rf_strategy_search import period_metrics
from random_forest_a_share_strategy import clean_for_json, split_dates


WALK_REPORT = REPORT_DIR_CSI / "walkforward"


def make_folds(first_90_dates: pd.DatetimeIndex) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    dates = pd.DatetimeIndex(first_90_dates)
    folds = []
    for train_frac, test_frac in [(0.50, 0.10), (0.60, 0.10), (0.70, 0.10), (0.80, 0.10)]:
        train_end = int(len(dates) * train_frac)
        test_end = int(len(dates) * (train_frac + test_frac))
        folds.append((dates[:train_end], dates[train_end:test_end]))
    return folds


def robust_score(metrics: list[dict[str, float]]) -> float:
    avg_sharpe = sum(item["sharpe"] for item in metrics) / len(metrics)
    min_sharpe = min(item["sharpe"] for item in metrics)
    avg_dd = sum(abs(item["max_drawdown"]) for item in metrics) / len(metrics)
    avg_win = sum(item["monthly_win_rate"] for item in metrics) / len(metrics)
    avg_positions = sum(item["avg_positions"] for item in metrics) / len(metrics)
    exposure_penalty = 0.5 if avg_positions < 2 else 0.0
    return avg_sharpe + 0.45 * min_sharpe + 0.25 * avg_win - 1.2 * max(0.0, avg_dd - 0.12) - exposure_penalty


def selected_candidates(limit: int = 80) -> pd.DataFrame:
    candidates = pd.read_csv(REPORT_DIR_CSI / "candidate_iterations.csv")
    top = candidates.sort_values("score", ascending=False).head(limit).copy()
    baseline = candidates[(candidates["model_id"] == 1) & (candidates["rule_id"] == 1)].copy()
    diversified = candidates[
        (candidates["validation_sharpe"] > 0.8)
        & (candidates["validation_max_drawdown"] > -0.20)
        & (candidates["validation_avg_positions"] >= 5)
    ].sort_values("validation_sharpe", ascending=False).head(30)
    out = pd.concat([top, baseline, diversified], ignore_index=True)
    return out.drop_duplicates(["model_id", "rule_id"])


def summarize_fold_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        "wf_avg_sharpe": float(sum(item["sharpe"] for item in metrics) / len(metrics)),
        "wf_min_sharpe": float(min(item["sharpe"] for item in metrics)),
        "wf_avg_max_drawdown": float(sum(item["max_drawdown"] for item in metrics) / len(metrics)),
        "wf_worst_max_drawdown": float(min(item["max_drawdown"] for item in metrics)),
        "wf_avg_monthly_win_rate": float(sum(item["monthly_win_rate"] for item in metrics) / len(metrics)),
        "wf_avg_positions": float(sum(item["avg_positions"] for item in metrics) / len(metrics)),
        "wf_score": robust_score(metrics),
    }


def main() -> None:
    WALK_REPORT.mkdir(parents=True, exist_ok=True)
    features = load_features().dropna(subset=["fwd_return_1"]).copy()
    train_dates, val_dates, live_dates = split_dates(features.dropna(subset=["ret_20", "vol_20", "ma_gap_20"]))
    first_90_dates = pd.DatetimeIndex(list(train_dates) + list(val_dates))
    folds = make_folds(first_90_dates)
    candidates = selected_candidates()

    model_by_id = {model.model_id: model for model in model_configs()}
    rule_by_id: dict[int, RuleConfig] = {rule.rule_id: rule for rule in rule_configs()}
    rows = []
    fold_artifacts = []

    for model_id, group in candidates.groupby("model_id"):
        model_config = model_by_id[int(model_id)]
        rules = [rule_by_id[int(rule_id)] for rule_id in group["rule_id"].tolist()]
        print(f"walk-forward model {model_id}: {model_config.change}", flush=True)
        rule_metrics: dict[int, list[dict[str, float]]] = {rule.rule_id: [] for rule in rules}

        for fold_num, (fold_train_dates, fold_test_dates) in enumerate(folds, start=1):
            fold_train = features[features["date"].isin(fold_train_dates)].copy()
            fold_test = features[features["date"].isin(fold_test_dates)].copy()
            model = fit_model(fold_train, model_config)
            predicted = predict(model, fold_test, model_config)
            for rule in rules:
                bt, _ = backtest(predicted, rule)
                metric = period_metrics(bt)
                rule_metrics[rule.rule_id].append(metric)
                fold_artifacts.append(
                    {
                        "model_id": int(model_id),
                        "rule_id": rule.rule_id,
                        "fold": fold_num,
                        **{f"fold_{key}": value for key, value in metric.items()},
                    }
                )

        for rule in rules:
            source = group[group["rule_id"] == rule.rule_id].iloc[0].to_dict()
            summary = summarize_fold_metrics(rule_metrics[rule.rule_id])
            rows.append(
                {
                    "model_id": int(model_id),
                    "rule_id": rule.rule_id,
                    "model_change": model_config.change,
                    "rule_change": rule.change,
                    **summary,
                    "source_validation_sharpe": source.get("validation_sharpe"),
                    "source_validation_max_drawdown": source.get("validation_max_drawdown"),
                    "source_paper_live_train_only_sharpe": source.get("paper_live_train_only_sharpe"),
                    **{f"model_{key}": value for key, value in model_config.__dict__.items()},
                    **{f"rule_{key}": value for key, value in rule.__dict__.items()},
                }
            )

    wf = pd.DataFrame(rows).sort_values("wf_score", ascending=False)
    wf.to_csv(WALK_REPORT / "walkforward_candidates.csv", index=False)
    pd.DataFrame(fold_artifacts).to_csv(WALK_REPORT / "walkforward_fold_metrics.csv", index=False)

    best = wf.iloc[0].to_dict()
    best_model = model_by_id[int(best["model_id"])]
    best_rule = rule_by_id[int(best["rule_id"])]
    final_bt, final_holdings, final_metrics = final_retrain_check(
        features, train_dates, val_dates, live_dates, best_model, best_rule
    )
    final_bt.to_csv(WALK_REPORT / "best_final_paper_live_equity.csv", index=False)
    final_holdings.to_csv(WALK_REPORT / "best_final_paper_live_holdings.csv", index=False)

    summary = {
        "method": "Select among prior CSI500 candidates by 4-fold expanding-window validation inside the first 90%; final 10% remains holdout.",
        "candidate_count": int(len(wf)),
        "best_walkforward_candidate": clean_for_json(best),
        "best_model": best_model.__dict__,
        "best_rule": best_rule.__dict__,
        "final_paper_live_retrained_on_90pct": final_metrics,
    }
    (WALK_REPORT / "summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "model_id",
        "rule_id",
        "wf_avg_sharpe",
        "wf_min_sharpe",
        "wf_worst_max_drawdown",
        "wf_avg_positions",
        "source_validation_sharpe",
        "source_paper_live_train_only_sharpe",
        "wf_score",
        "model_change",
        "rule_change",
    ]
    print(wf[display_cols].head(20).to_string(index=False))
    print("\nBest model:", best_model)
    print("Best rule:", best_rule)
    print("Final paper-live metrics:")
    print(json.dumps(clean_for_json(final_metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
