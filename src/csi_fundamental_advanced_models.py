from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from csi_fundamental_improvement_analysis import (
    REPORT_BASE,
    attach_industry,
    diversified_backtest,
    diversified_rules,
    summarize_equity,
)
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_adaptive_walkforward import block_bootstrap, make_folds
from csi_medium_term_strategy import metrics
from random_forest_a_share_strategy import clean_for_json


@dataclass(frozen=True)
class AdvancedSpec:
    model_id: int
    feature_set: str
    change: str
    target: str = "target_mix_20_40_60"
    n_estimators: int = 300
    max_depth: int | None = 5
    min_samples_leaf: int = 80
    max_features: str | float = "sqrt"
    max_samples: float = 0.45
    random_state: int = 42


STRUCTURED_BASE = [
    "f_value",
    "f_quality",
    "f_growth",
    "f_low_debt",
    "f_cashflow",
    "f_momentum",
    "f_lowvol",
    "f_small",
    "f_value_quality",
    "f_growth_quality",
    "market_ret_20",
    "market_ret_60",
    "market_breadth_20",
]


RAW_VALUE_FIN = [
    "val_log_total_mv_rank",
    "val_log_total_mv",
    "val_total_mv",
    "val_log_float_mv_rank",
    "val_book_to_price_rank",
    "val_sales_to_price_rank",
    "val_earnings_yield_rank",
    "val_pb",
    "val_pe_ttm",
    "val_peg",
    "fin_roe",
    "fin_roe_deducted",
    "fin_parent_profit",
    "fin_deduct_profit",
    "fin_ocf_per_share",
    "fin_debt_to_assets",
    "fin_revenue_yoy",
    "fin_parent_profit_yoy",
    "fin_deduct_profit_yoy",
    "fin_single_q_revenue_yoy",
    "fin_single_q_profit_yoy",
    "market_ret_20",
    "market_ret_60",
    "market_breadth_20",
]


def rank_pct(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df.groupby("date")[col].rank(pct=True, ascending=ascending)


def mean_existing(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1)


def add_mixed_target(features: pd.DataFrame) -> pd.DataFrame:
    out = features.sort_values(["symbol", "date"]).copy()
    pieces = []
    for _, group in out.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        if "fwd_return_60" not in group.columns:
            group["fwd_return_60"] = group["close"].pct_change(60).shift(-60)
        pieces.append(group)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    if "target_rank_20" not in out.columns:
        out["target_rank_20"] = out.groupby("date")["fwd_return_20"].rank(pct=True)
    if "target_rank_40" not in out.columns:
        out["target_rank_40"] = out.groupby("date")["fwd_return_40"].rank(pct=True)
    out["target_rank_60"] = out.groupby("date")["fwd_return_60"].rank(pct=True)
    out["target_mix_20_40_60"] = out[["target_rank_20", "target_rank_40", "target_rank_60"]].mean(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def add_structured_features(features: pd.DataFrame) -> pd.DataFrame:
    out = add_mixed_target(features)
    out["rank_bp"] = rank_pct(out, "val_book_to_price", True)
    out["rank_sp"] = rank_pct(out, "val_sales_to_price", True)
    out["rank_ep"] = rank_pct(out, "val_earnings_yield", True)
    out["rank_low_pb"] = rank_pct(out, "val_pb", False)
    out["rank_low_pe"] = rank_pct(out, "val_pe_ttm", False)
    out["rank_low_debt"] = rank_pct(out, "fin_debt_to_assets", False)
    out["rank_roe"] = rank_pct(out, "fin_roe", True)
    out["rank_roic"] = rank_pct(out, "fin_roic", True)
    out["rank_ocfps"] = rank_pct(out, "fin_ocf_per_share", True)
    out["rank_gross_margin"] = rank_pct(out, "fin_gross_margin", True)
    out["rank_revenue_yoy"] = rank_pct(out, "fin_revenue_yoy", True)
    out["rank_profit_yoy"] = rank_pct(out, "fin_parent_profit_yoy", True)
    out["rank_deduct_profit_yoy"] = rank_pct(out, "fin_deduct_profit_yoy", True)
    out["rank_single_q_revenue_yoy"] = rank_pct(out, "fin_single_q_revenue_yoy", True)
    out["rank_single_q_profit_yoy"] = rank_pct(out, "fin_single_q_profit_yoy", True)
    out["rank_ret60"] = rank_pct(out, "ret_60", True)
    out["rank_rel_ret60"] = rank_pct(out, "rel_ret_60", True)
    out["rank_low_vol20"] = rank_pct(out, "vol_20", False)
    out["rank_small"] = rank_pct(out, "val_log_total_mv", False)

    out["f_value"] = mean_existing(out, ["rank_bp", "rank_sp", "rank_ep", "rank_low_pb", "rank_low_pe"])
    out["f_quality"] = mean_existing(out, ["rank_roe", "rank_roic", "rank_gross_margin"])
    out["f_growth"] = mean_existing(out, ["rank_revenue_yoy", "rank_profit_yoy", "rank_deduct_profit_yoy", "rank_single_q_revenue_yoy", "rank_single_q_profit_yoy"])
    out["f_low_debt"] = out["rank_low_debt"]
    out["f_cashflow"] = out["rank_ocfps"]
    out["f_momentum"] = mean_existing(out, ["rank_ret60", "rank_rel_ret60"])
    out["f_lowvol"] = out["rank_low_vol20"]
    out["f_small"] = out["rank_small"]
    out["f_value_quality"] = mean_existing(out, ["f_value", "f_quality", "f_low_debt", "f_cashflow"])
    out["f_growth_quality"] = mean_existing(out, ["f_growth", "f_quality", "f_cashflow"])

    neutral_cols = [col for col in STRUCTURED_BASE if col.startswith("f_")]
    for col in neutral_cols:
        out[f"{col}_industry_rank"] = out.groupby(["date", "industry_gate"])[col].rank(pct=True)
    out["size_bucket"] = out.groupby("date")["val_log_total_mv_rank"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=min(5, max(1, s.notna().sum())), labels=False, duplicates="drop")
        if s.notna().sum() >= 5
        else np.nan
    )
    for col in neutral_cols:
        out[f"{col}_size_rank"] = out.groupby(["date", "size_bucket"])[col].rank(pct=True)
    return out.replace([np.inf, -np.inf], np.nan)


def advanced_specs() -> list[AdvancedSpec]:
    return [
        AdvancedSpec(1, "structured", "结构化价值/质量/成长/动量因子 + 20/40/60混合目标。"),
        AdvancedSpec(2, "structured_no_size", "结构化因子但去掉小市值因子，测试非小盘alpha。"),
        AdvancedSpec(3, "industry_neutral", "行业内rank结构化因子，测试行业中性alpha。"),
        AdvancedSpec(4, "size_neutral", "市值分组内rank结构化因子，测试市值中性alpha。"),
        AdvancedSpec(5, "raw_value_fin_mixed", "原始估值+财报特征，但目标改为20/40/60混合rank。", max_depth=6, min_samples_leaf=80),
    ]


def columns_for_spec(spec: AdvancedSpec, features: pd.DataFrame) -> list[str]:
    if spec.feature_set == "structured":
        cols = STRUCTURED_BASE
    elif spec.feature_set == "structured_no_size":
        cols = [col for col in STRUCTURED_BASE if col != "f_small"]
    elif spec.feature_set == "industry_neutral":
        cols = [f"{col}_industry_rank" for col in STRUCTURED_BASE if col.startswith("f_")] + ["market_ret_20", "market_ret_60", "market_breadth_20"]
    elif spec.feature_set == "size_neutral":
        cols = [f"{col}_size_rank" for col in STRUCTURED_BASE if col.startswith("f_")] + ["market_ret_20", "market_ret_60", "market_breadth_20"]
    elif spec.feature_set == "raw_value_fin_mixed":
        cols = RAW_VALUE_FIN
    else:
        raise ValueError(spec.feature_set)
    return [col for col in dict.fromkeys(cols) if col in features.columns]


def fit_model(train: pd.DataFrame, spec: AdvancedSpec, cols: list[str]) -> Pipeline:
    sample = train.dropna(subset=[spec.target]).copy()
    model = RandomForestRegressor(
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        max_features=spec.max_features,
        max_samples=spec.max_samples,
        bootstrap=True,
        random_state=spec.random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    pipe.fit(sample[cols], sample[spec.target])
    return pipe


def predict(model: Pipeline, frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[cols])
    return out


def score_validation(m: dict[str, float]) -> float:
    if m.get("avg_positions", 0.0) < 1.0 or m.get("avg_exposure", 0.0) < 0.05:
        return -10.0
    position_penalty = max(0.0, 5.0 - m.get("avg_positions", 0.0)) * 0.20
    exposure_penalty = max(0.0, 0.25 - m.get("avg_exposure", 0.0)) * 1.5
    dd_penalty = max(0.0, abs(m.get("max_drawdown", 0.0)) - 0.18) * 1.5
    return m["sharpe"] + 0.20 * m.get("win_rate", 0.0) - position_penalty - exposure_penalty - dd_penalty


def select_candidate(train: pd.DataFrame, val: pd.DataFrame, features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    for spec in advanced_specs():
        cols = columns_for_spec(spec, features)
        model = fit_model(train, spec, cols)
        val_pred = predict(model, val, cols)
        for rule in diversified_rules(constrained_only=True):
            val_bt, _ = diversified_backtest(val_pred, rule)
            val_m = metrics(val_bt)
            row = {
                "model_id": spec.model_id,
                "feature_set": spec.feature_set,
                "feature_count": len(cols),
                "rule_id": rule.rule_id,
                "score": score_validation(val_m),
                "model_change": spec.change,
                "rule_change": rule.change,
                **{f"validation_{key}": value for key, value in val_m.items()},
            }
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = {**row, "model": spec, "rule": rule, "cols": cols}
    return best, pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"advanced_n{args.max_symbols}_{args.report_name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    features = load_cached_features(args.max_symbols)
    features = attach_industry(features, Path(args.industry_map), args.industry_col)
    features = add_structured_features(features)
    eligible = features.dropna(subset=["ret_60", "fwd_return_40", "fwd_return_60"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    fold_rows = []
    selection_rows = []
    equity_frames = []
    holdings_frames = []
    importance_rows = []
    for fold_id, (train_dates, val_dates, test_dates) in enumerate(folds, start=1):
        print(
            f"[fold {fold_id}/{len(folds)}] train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"val {val_dates[0].date()}..{val_dates[-1].date()} test {test_dates[0].date()}..{test_dates[-1].date()}",
            flush=True,
        )
        train = features[features["date"].isin(train_dates)].copy()
        val = features[features["date"].isin(val_dates)].copy()
        test = features[features["date"].isin(test_dates)].copy()
        best, selection = select_candidate(train, val, features)
        selection["fold_id"] = fold_id
        selection_rows.append(selection)
        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        model = fit_model(train_full, best["model"], best["cols"])
        pred = predict(model, test, best["cols"])
        bt, holdings = diversified_backtest(pred, best["rule"])
        bt["fold_id"] = fold_id
        holdings["fold_id"] = fold_id
        equity_frames.append(bt)
        holdings_frames.append(holdings)
        m = metrics(bt)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "selected_model_id": best["model"].model_id,
                "selected_feature_set": best["model"].feature_set,
                "selected_feature_count": len(best["cols"]),
                "selected_rule_id": best["rule"].rule_id,
                "validation_score": best["score"],
                "validation_sharpe": best["validation_sharpe"],
                **{f"test_{key}": value for key, value in m.items()},
            }
        )
        importance_rows.append(
            pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "feature": best["cols"],
                    "importance": model.named_steps["model"].feature_importances_,
                    "feature_set": best["model"].feature_set,
                }
            )
        )

    folds_df = pd.DataFrame(fold_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    equity = pd.concat(equity_frames, ignore_index=True)
    holdings = pd.concat(holdings_frames, ignore_index=True)
    equity["combined_equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    feature_importance = (
        pd.concat(importance_rows, ignore_index=True)
        .groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), selected_folds=("fold_id", "nunique"))
        .sort_values("mean_importance", ascending=False)
    )

    folds_df.to_csv(report_dir / "advanced_walkforward_folds.csv", index=False)
    selections.to_csv(report_dir / "selection_candidates_by_fold.csv", index=False)
    equity.to_csv(report_dir / "advanced_walkforward_equity.csv", index=False)
    holdings.to_csv(report_dir / "advanced_walkforward_holdings.csv", index=False)
    feature_importance.to_csv(report_dir / "feature_importance.csv", index=False)

    combined = summarize_equity(equity, folds_df, args)
    returns = equity["net_return"].fillna(0)
    summary = {
        "objective": "Advanced tests: 200-stock expansion, mixed 20/40/60 target, structured factors, industry/size-neutral pressure variants.",
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
        "walkforward": combined,
        "selected_feature_sets": clean_for_json(folds_df["selected_feature_set"].value_counts().to_dict()),
        "selected_rules": clean_for_json(folds_df["selected_rule_id"].value_counts().to_dict()),
        "top_features": clean_for_json(feature_importance.head(25).to_dict(orient="records")),
        "bootstrap_sharpe": block_bootstrap(returns, args.bootstrap, args.block_len, args.seed),
    }
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAdvanced folds:")
    print(
        folds_df[
            [
                "fold_id",
                "selected_feature_set",
                "selected_rule_id",
                "validation_sharpe",
                "test_sharpe",
                "test_annual_return",
                "test_max_drawdown",
                "test_avg_positions",
                "test_avg_exposure",
            ]
        ].to_string(index=False)
    )
    print("\nSummary:")
    print(json.dumps(clean_for_json(summary["walkforward"]), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=200)
    parser.add_argument("--report-name", default="v1")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=194)
    parser.add_argument("--test-len", type=int, default=194)
    parser.add_argument("--embargo", type=int, default=60)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--industry-map", default="data/processed/csi1000_industry_map_200.csv")
    parser.add_argument("--industry-col", default="industry_gate")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
