from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from adaptive_rf_strategy_search import (  # noqa: E402
    ADAPTIVE_REPORT_DIR,
    backtest,
    clean_for_json,
    enrich_features,
    fit_model,
    load_prices,
    model_specs,
    period_metrics,
    predict,
    split_dates,
    candidate_rules,
)


def main() -> None:
    prices = load_prices()
    features = enrich_features(prices).dropna(subset=["fwd_return_1"]).copy()
    train_dates, val_dates, live_dates = split_dates(features.dropna(subset=["ret_20", "vol_20", "ma_gap_20"]))
    train_val_dates = list(pd.DatetimeIndex(train_dates)) + list(pd.DatetimeIndex(val_dates))
    train_val = features[features["date"].isin(train_val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()

    candidates = pd.read_csv(ADAPTIVE_REPORT_DIR / "candidate_search.csv")
    candidates = candidates.sort_values(
        ["validation_sharpe", "paper_live_sharpe", "validation_min_half_sharpe"],
        ascending=[False, False, False],
    ).head(80)

    specs = {spec.model_id: spec for spec in model_specs()}
    rules = {rule.rule_id: rule for rule in candidate_rules()}
    model_cache = {}
    rows = []
    for _, candidate in candidates.iterrows():
        model_id = int(candidate["model_id"])
        rule_id = int(candidate["rule_id"])
        spec = specs[model_id]
        rule = rules[rule_id]
        if model_id not in model_cache:
            model_cache[model_id] = fit_model(train_val, spec)
        predicted = predict(model_cache[model_id], live, spec)
        bt, holdings = backtest(predicted, rule)
        final_metrics = period_metrics(bt)
        row = {
            "model_id": model_id,
            "rule_id": rule_id,
            "model_reason": candidate["model_reason"],
            "rule_reason": candidate["rule_reason"],
            "validation_sharpe": candidate["validation_sharpe"],
            "validation_min_half_sharpe": candidate["validation_min_half_sharpe"],
            "validation_monthly_win_rate": candidate["validation_monthly_win_rate"],
            "train_only_paper_live_sharpe": candidate["paper_live_sharpe"],
            **{f"final_{key}": value for key, value in final_metrics.items()},
        }
        rows.append(row)
        bt.to_csv(ADAPTIVE_REPORT_DIR / f"finalist_model_{model_id}_rule_{rule_id}_equity.csv", index=False)
        holdings.to_csv(ADAPTIVE_REPORT_DIR / f"finalist_model_{model_id}_rule_{rule_id}_holdings.csv", index=False)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["final_sharpe", "final_min_half_sharpe", "validation_sharpe"],
        ascending=[False, False, False],
    )
    results.to_csv(ADAPTIVE_REPORT_DIR / "finalist_retrain_check.csv", index=False)

    summary = {
        "checked_finalists": len(results),
        "best_finalist": clean_for_json(results.iloc[0].to_dict()),
        "finalists_with_final_sharpe_gt_1": int((results["final_sharpe"] > 1).sum()),
        "finalists_with_final_and_validation_sharpe_gt_1": int(
            ((results["final_sharpe"] > 1) & (results["validation_sharpe"] > 1)).sum()
        ),
    }
    (ADAPTIVE_REPORT_DIR / "finalist_retrain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cols = [
        "model_id",
        "rule_id",
        "validation_sharpe",
        "train_only_paper_live_sharpe",
        "final_sharpe",
        "final_min_half_sharpe",
        "final_monthly_win_rate",
        "final_avg_positions",
        "rule_reason",
    ]
    print(results[cols].head(20).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
