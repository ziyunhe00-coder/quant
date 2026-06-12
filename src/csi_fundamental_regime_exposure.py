from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from csi_fundamental_improvement_analysis import (
    attach_industry,
    columns_for_spec,
    fit_model,
    model_specs,
    predict_frame,
    score_validation,
    select_diversified_weights,
)
from csi_fundamental_rf_small import REPORT_BASE
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_adaptive_walkforward import block_bootstrap, make_folds
from csi_medium_term_strategy import metrics, turnover
from csi_fundamental_improvement_analysis import diversified_rules
from random_forest_a_share_strategy import clean_for_json


@dataclass(frozen=True)
class ExposureRule:
    exposure_id: int
    name: str
    change: str
    mode: str
    min_exposure: float = 0.15
    max_exposure: float = 0.85
    target_vol: float = 0.18
    weak_exposure: float = 0.20
    neutral_exposure: float = 0.45
    strong_exposure: float = 0.80
    dd_cut_1: float = -0.08
    dd_cut_2: float = -0.14


def exposure_rules(exposure_set: str = "all") -> list[ExposureRule]:
    rules = [
        ExposureRule(1, "fixed", "固定暴露：沿用规则自身总暴露上限。", "fixed"),
        ExposureRule(2, "trend_breadth", "趋势/宽度分档：强市高仓、弱市低仓。", "trend_breadth"),
        ExposureRule(3, "vol_target", "波动率目标：按市场20日年化波动调整暴露。", "vol_target", target_vol=0.18),
        ExposureRule(4, "trend_vol", "趋势宽度 + 波动率目标组合。", "trend_vol", target_vol=0.18),
        ExposureRule(5, "trend_vol_dd", "趋势宽度 + 波动率目标 + 组合回撤降仓。", "trend_vol_dd", target_vol=0.18),
        ExposureRule(6, "conservative_dd", "保守仓位：趋势分档并在回撤时强制降仓。", "trend_breadth_dd", max_exposure=0.70),
    ]
    balanced = [
        ExposureRule(21, "balanced_trend", "均衡趋势/宽度：弱市保留三成左右仓位，强市接近规则上限。", "trend_breadth", 0.25, 0.85, 0.24, 0.30, 0.60, 0.85),
        ExposureRule(22, "balanced_vol", "均衡波动率目标：目标年化波动提高到24%。", "vol_target", 0.25, 0.85, 0.24, 0.30, 0.60, 0.85),
        ExposureRule(23, "balanced_trend_vol", "均衡趋势 + 波动率目标，保留风险预算约束。", "trend_vol", 0.25, 0.85, 0.24, 0.30, 0.60, 0.85),
        ExposureRule(24, "balanced_trend_vol_dd", "均衡趋势 + 波动率目标 + 回撤降仓。", "trend_vol_dd", 0.25, 0.85, 0.24, 0.30, 0.60, 0.85),
    ]
    growth = [
        ExposureRule(31, "growth_trend", "进攻趋势/宽度：弱市不低于35%，强市满规则上限。", "trend_breadth", 0.30, 0.90, 0.30, 0.35, 0.70, 0.90),
        ExposureRule(32, "growth_vol", "进攻波动率目标：目标年化波动提高到30%。", "vol_target", 0.30, 0.90, 0.30, 0.35, 0.70, 0.90),
        ExposureRule(33, "growth_trend_vol", "进攻趋势 + 波动率目标。", "trend_vol", 0.30, 0.90, 0.30, 0.35, 0.70, 0.90),
        ExposureRule(34, "growth_trend_vol_dd", "进攻趋势 + 波动率目标 + 回撤降仓。", "trend_vol_dd", 0.30, 0.90, 0.30, 0.35, 0.70, 0.90),
    ]
    if exposure_set == "fixed":
        return [rule for rule in rules if rule.name == "fixed"]
    if exposure_set == "dynamic":
        return [rule for rule in rules if rule.name != "fixed"]
    if exposure_set == "balanced":
        return balanced
    if exposure_set == "growth":
        return growth
    if exposure_set == "return_seek":
        return balanced + growth
    return rules


def trend_exposure(day: pd.DataFrame, rule: ExposureRule) -> float:
    first = day.iloc[0]
    ret20 = float(first.get("market_ret_20", 0.0))
    ret60 = float(first.get("market_ret_60", 0.0))
    breadth = float(first.get("market_breadth_20", 0.0))
    if ret60 > 0.04 and ret20 > -0.02 and breadth > 0.55:
        return rule.strong_exposure
    if ret60 > -0.03 and ret20 > -0.06 and breadth > 0.40:
        return rule.neutral_exposure
    if ret20 < -0.08 or ret60 < -0.08 or breadth < 0.28:
        return rule.weak_exposure
    return max(rule.min_exposure, min(rule.max_exposure, 0.32))


def vol_exposure(day: pd.DataFrame, rule: ExposureRule) -> float:
    vol = float(day.iloc[0].get("market_vol_20", np.nan))
    if not np.isfinite(vol) or vol <= 0:
        return rule.neutral_exposure
    return float(np.clip(rule.target_vol / vol, rule.min_exposure, rule.max_exposure))


def drawdown_multiplier(drawdown: float, rule: ExposureRule) -> float:
    if drawdown <= rule.dd_cut_2:
        return 0.35
    if drawdown <= rule.dd_cut_1:
        return 0.60
    return 1.0


def target_exposure(day: pd.DataFrame, rule: ExposureRule, portfolio_drawdown: float, base_max: float) -> float:
    if rule.mode == "fixed":
        target = base_max
    elif rule.mode == "trend_breadth":
        target = trend_exposure(day, rule)
    elif rule.mode == "vol_target":
        target = vol_exposure(day, rule)
    elif rule.mode == "trend_vol":
        target = min(trend_exposure(day, rule), vol_exposure(day, rule))
    elif rule.mode == "trend_vol_dd":
        target = min(trend_exposure(day, rule), vol_exposure(day, rule)) * drawdown_multiplier(portfolio_drawdown, rule)
    elif rule.mode == "trend_breadth_dd":
        target = trend_exposure(day, rule) * drawdown_multiplier(portfolio_drawdown, rule)
    else:
        raise ValueError(rule.mode)
    return float(np.clip(target, 0.0, min(base_max, rule.max_exposure)))


def scale_weights(weights: dict[str, float], target_gross: float) -> dict[str, float]:
    gross = sum(abs(value) for value in weights.values())
    if gross <= 0:
        return {}
    scale = target_gross / gross
    if scale >= 0.999 and gross <= target_gross:
        return weights
    return {symbol: weight * scale for symbol, weight in weights.items()}


def dynamic_backtest(predicted: pd.DataFrame, trade_rule, exposure_rule: ExposureRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings_rows = []
    weights: dict[str, float] = {}
    entry: dict[str, float] = {}
    peak: dict[str, float] = {}
    age: dict[str, int] = {}
    equity = 1.0
    equity_peak = 1.0

    for i, (date, day) in enumerate(predicted.groupby("date", sort=True)):
        day = day.copy()
        close_by_symbol = day.set_index("symbol")["close"]
        return_by_symbol = day.set_index("symbol")["fwd_return_1"]

        stopped = []
        for symbol in list(weights):
            close = close_by_symbol.get(symbol, np.nan)
            if pd.isna(close):
                stopped.append(symbol)
                continue
            peak[symbol] = max(peak.get(symbol, close), float(close))
            pnl = float(close) / entry[symbol] - 1
            dd_from_peak = float(close) / peak[symbol] - 1
            age[symbol] = age.get(symbol, 0) + 1
            if pnl <= trade_rule.hard_stop or dd_from_peak <= trade_rule.trailing_stop or age[symbol] >= trade_rule.max_holding_days:
                stopped.append(symbol)
        for symbol in stopped:
            weights.pop(symbol, None)
            entry.pop(symbol, None)
            peak.pop(symbol, None)
            age.pop(symbol, None)

        target = weights.copy()
        if i % trade_rule.rebalance_every == 0:
            target = select_diversified_weights(day, trade_rule)
            for symbol in target:
                close = close_by_symbol.get(symbol, np.nan)
                if symbol not in entry and pd.notna(close):
                    entry[symbol] = float(close)
                    peak[symbol] = float(close)
                    age[symbol] = 0
            for symbol in list(entry):
                if symbol not in target:
                    entry.pop(symbol, None)
                    peak.pop(symbol, None)
                    age.pop(symbol, None)

        portfolio_drawdown = equity / equity_peak - 1
        desired_exposure = target_exposure(day, exposure_rule, portfolio_drawdown, trade_rule.max_gross_exposure)
        target = scale_weights(target, desired_exposure)
        for symbol in list(entry):
            if symbol not in target:
                entry.pop(symbol, None)
                peak.pop(symbol, None)
                age.pop(symbol, None)

        day_turnover = turnover(weights, target)
        weights = target
        gross_return = 0.0
        for symbol, weight in weights.items():
            item_return = return_by_symbol.get(symbol, np.nan)
            if pd.notna(item_return):
                contribution = weight * float(item_return)
                gross_return += contribution
                holdings_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "weight": weight,
                        "age": age.get(symbol, 0),
                        "fwd_return_1": item_return,
                        "contribution": contribution,
                        "target_exposure": desired_exposure,
                        "exposure_rule": exposure_rule.name,
                    }
                )
        cost = day_turnover * trade_rule.transaction_cost_bps / 10000
        net_return = gross_return - cost
        equity *= 1 + net_return
        equity_peak = max(equity_peak, equity)

        if weights and 1 + net_return != 0:
            weights = {
                symbol: weight * (1 + float(return_by_symbol.get(symbol, 0.0))) / (1 + net_return)
                for symbol, weight in weights.items()
            }
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "equity": equity,
                "n_positions": len(weights),
                "gross_exposure": sum(abs(w) for w in weights.values()),
                "target_exposure": desired_exposure,
                "turnover": day_turnover,
                "stops": len(stopped),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(holdings_rows)


def regime_score(m: dict[str, float], score_mode: str = "sharpe") -> float:
    if m.get("avg_positions", 0.0) < 1.0 or m.get("avg_exposure", 0.0) < 0.05:
        return -10.0
    dd_penalty = max(0.0, abs(m.get("max_drawdown", 0.0)) - 0.16) * 2.0
    if score_mode == "return_balance":
        low_return_penalty = max(0.0, 0.10 - m.get("annual_return", 0.0)) * 2.5
        low_exposure_penalty = max(0.0, 0.22 - m.get("avg_exposure", 0.0)) * 1.5
        return (
            m["sharpe"]
            + 0.30 * m.get("annual_return", 0.0)
            + 0.10 * m.get("win_rate", 0.0)
            - dd_penalty
            - low_return_penalty
            - low_exposure_penalty
        )
    low_return_penalty = max(0.0, 0.04 - m.get("annual_return", 0.0)) * 2.0
    return m["sharpe"] + 0.15 * m.get("win_rate", 0.0) - dd_penalty - low_return_penalty


def select_candidate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: pd.DataFrame,
    exp_rules: list[ExposureRule],
    score_mode: str,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    for spec in model_specs(features):
        cols = columns_for_spec(features, spec)
        model = fit_model(train, spec, cols)
        val_pred = predict_frame(model, val, cols)
        for trade_rule in diversified_rules(constrained_only=True):
            for exp_rule in exp_rules:
                val_bt, _ = dynamic_backtest(val_pred, trade_rule, exp_rule)
                val_m = metrics(val_bt)
                row = {
                    "model_id": spec.model_id,
                    "feature_set": spec.feature_set,
                    "feature_count": len(cols),
                    "trade_rule_id": trade_rule.rule_id,
                    "exposure_id": exp_rule.exposure_id,
                    "exposure_name": exp_rule.name,
                    "score": regime_score(val_m, score_mode),
                    "model_change": spec.change,
                    "trade_rule_change": trade_rule.change,
                    "exposure_change": exp_rule.change,
                    **{f"validation_{key}": value for key, value in val_m.items()},
                }
                rows.append(row)
                if best is None or row["score"] > best["score"]:
                    best = {**row, "model": spec, "cols": cols, "trade_rule": trade_rule, "exposure_rule": exp_rule}
    return best, pd.DataFrame(rows)


def combined_metrics(equity: pd.DataFrame) -> dict[str, float]:
    out = equity.copy()
    out["equity"] = (1 + out["net_return"].fillna(0)).cumprod()
    return metrics(out)


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"regime_n{args.max_symbols}_{args.report_name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    features = load_cached_features(args.max_symbols)
    features = attach_industry(features, Path(args.industry_map), args.industry_col)
    exp_rules = exposure_rules(args.exposure_set)
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)

    fold_rows = []
    selection_rows = []
    equity_frames = []
    holdings_frames = []
    for fold_id, (train_dates, val_dates, test_dates) in enumerate(folds, start=1):
        print(
            f"[fold {fold_id}/{len(folds)}] train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"val {val_dates[0].date()}..{val_dates[-1].date()} test {test_dates[0].date()}..{test_dates[-1].date()}",
            flush=True,
        )
        train = features[features["date"].isin(train_dates)].copy()
        val = features[features["date"].isin(val_dates)].copy()
        test = features[features["date"].isin(test_dates)].copy()
        best, selection = select_candidate(train, val, features, exp_rules, args.score_mode)
        selection["fold_id"] = fold_id
        selection_rows.append(selection)

        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        model = fit_model(train_full, best["model"], best["cols"])
        pred = predict_frame(model, test, best["cols"])
        bt, holdings = dynamic_backtest(pred, best["trade_rule"], best["exposure_rule"])
        bt["fold_id"] = fold_id
        holdings["fold_id"] = fold_id
        equity_frames.append(bt)
        holdings_frames.append(holdings)
        test_m = metrics(bt)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "selected_model_id": best["model"].model_id,
                "selected_feature_set": best["model"].feature_set,
                "selected_trade_rule_id": best["trade_rule"].rule_id,
                "selected_exposure_id": best["exposure_rule"].exposure_id,
                "selected_exposure_name": best["exposure_rule"].name,
                "validation_score": best["score"],
                "validation_sharpe": best["validation_sharpe"],
                **{f"test_{key}": value for key, value in test_m.items()},
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    equity = pd.concat(equity_frames, ignore_index=True)
    holdings = pd.concat(holdings_frames, ignore_index=True)
    equity["combined_equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    combined = combined_metrics(equity)
    returns = equity["net_return"].fillna(0)
    summary = {
        "objective": "Dynamic exposure overlays: trend/breadth, volatility targeting, drawdown control.",
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
            "exposure_set": args.exposure_set,
            "score_mode": args.score_mode,
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
        "avg_target_exposure": float(equity["target_exposure"].mean()) if "target_exposure" in equity else None,
    }
    folds_df.to_csv(report_dir / "regime_walkforward_folds.csv", index=False)
    selections.to_csv(report_dir / "selection_candidates_by_fold.csv", index=False)
    equity.to_csv(report_dir / "regime_walkforward_equity.csv", index=False)
    holdings.to_csv(report_dir / "regime_walkforward_holdings.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRegime folds:")
    print(
        folds_df[
            [
                "fold_id",
                "selected_feature_set",
                "selected_trade_rule_id",
                "selected_exposure_name",
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
    parser.add_argument("--industry-map", default="data/processed/csi1000_industry_map_200.csv")
    parser.add_argument("--industry-col", default="industry_gate")
    parser.add_argument("--exposure-set", choices=["all", "fixed", "dynamic", "balanced", "growth", "return_seek"], default="all")
    parser.add_argument("--score-mode", choices=["sharpe", "return_balance"], default="sharpe")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
