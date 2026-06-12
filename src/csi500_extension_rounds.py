from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rf_strategy_search import period_metrics
from csi500_iterative_improvement import (
    CORE_FEATURES,
    ModelConfig,
    fit_model,
    load_features,
    predict,
    target_col,
)
from random_forest_a_share_strategy import REPORT_DIR, clean_for_json, split_dates


EXT_REPORT = REPORT_DIR / "csi500_extensions"


@dataclass(frozen=True)
class ModelBundle:
    bundle_id: int
    round_id: int
    change: str
    combine: str
    models: tuple[ModelConfig, ...]


@dataclass(frozen=True)
class ExtensionRule:
    rule_id: int
    round_id: int
    change: str
    top_n: int
    rebalance_every: int
    market_filter: str
    stock_filter: str
    weighting: str
    pred_quantile: float
    exposure_mode: str
    max_weight: float | None
    risk_stop: str
    transaction_cost_bps: float = 12.0


def model_bundles() -> list[ModelBundle]:
    rank_1 = ModelConfig(
        model_id=31,
        iteration=1,
        change="上一轮最佳结构：预测下一日横截面排名。",
        target="rank_1",
        features=CORE_FEATURES,
        n_estimators=160,
        max_depth=5,
        min_samples_leaf=90,
        max_samples=0.35,
    )
    rank_5 = ModelConfig(
        model_id=51,
        iteration=5,
        change="扩展：预测5日横截面排名，观察中期排序信号。",
        target="rank_5",
        features=CORE_FEATURES,
        n_estimators=140,
        max_depth=5,
        min_samples_leaf=100,
        max_samples=0.35,
    )
    excess_1 = ModelConfig(
        model_id=21,
        iteration=5,
        change="扩展：预测下一日超额收益，降低市场 beta 污染。",
        target="excess_1",
        features=CORE_FEATURES,
        n_estimators=140,
        max_depth=5,
        min_samples_leaf=100,
        max_samples=0.35,
    )
    return [
        ModelBundle(1, 1, "基准模型：下一日横截面排名。", "single", (rank_1,)),
        ModelBundle(2, 5, "模型集成：rank_1 与 rank_5 的日内预测排名均值。", "mean_rank", (rank_1, rank_5)),
        ModelBundle(3, 5, "模型集成：rank_1、rank_5、excess_1 的日内预测排名均值。", "mean_rank", (rank_1, rank_5, excess_1)),
    ]


def extension_rules() -> list[ExtensionRule]:
    rules: list[ExtensionRule] = []
    rule_id = 1

    def add(
        round_id,
        change,
        top_ns,
        rebalances,
        market_filters,
        stock_filters,
        weightings,
        quantiles,
        exposure_modes,
        max_weights,
        risk_stops,
    ):
        nonlocal rule_id
        for top_n in top_ns:
            for rebalance_every in rebalances:
                for market_filter in market_filters:
                    for stock_filter in stock_filters:
                        for weighting in weightings:
                            for pred_quantile in quantiles:
                                for exposure_mode in exposure_modes:
                                    for max_weight in max_weights:
                                        for risk_stop in risk_stops:
                                            rules.append(
                                                ExtensionRule(
                                                    rule_id=rule_id,
                                                    round_id=round_id,
                                                    change=change,
                                                    top_n=top_n,
                                                    rebalance_every=rebalance_every,
                                                    market_filter=market_filter,
                                                    stock_filter=stock_filter,
                                                    weighting=weighting,
                                                    pred_quantile=pred_quantile,
                                                    exposure_mode=exposure_mode,
                                                    max_weight=max_weight,
                                                    risk_stop=risk_stop,
                                                )
                                            )
                                            rule_id += 1

    add(
        round_id=1,
        change="复现上一轮 walk-forward 策略。",
        top_ns=[10],
        rebalances=[10],
        market_filters=["breadth_gt_60"],
        stock_filters=["above_ma20"],
        weightings=["inverse_vol"],
        quantiles=[0.90],
        exposure_modes=["full"],
        max_weights=[None],
        risk_stops=["none"],
    )
    add(
        round_id=2,
        change="分散度扩展：提高持仓数量并测试更慢调仓。",
        top_ns=[10, 15, 20, 30],
        rebalances=[10, 15, 20],
        market_filters=["breadth_gt_60"],
        stock_filters=["above_ma20"],
        weightings=["inverse_vol", "equal"],
        quantiles=[0.85, 0.90],
        exposure_modes=["full"],
        max_weights=[None, 0.15],
        risk_stops=["none"],
    )
    add(
        round_id=3,
        change="过滤扩展：加入流动性、低波动和更严格市场状态。",
        top_ns=[10, 15, 20],
        rebalances=[10, 20],
        market_filters=["breadth_gt_60", "breadth_gt_65", "risk_on_60", "trend_20_60"],
        stock_filters=["above_ma20", "above_ma60", "liquid_above_median", "liquid_low_vol"],
        weightings=["inverse_vol"],
        quantiles=[0.85, 0.90, 0.95],
        exposure_modes=["full"],
        max_weights=[0.15, 0.20],
        risk_stops=["none"],
    )
    add(
        round_id=4,
        change="风险扩展：波动目标、宽度缩放和回撤冷却。",
        top_ns=[10, 15, 20],
        rebalances=[10, 20],
        market_filters=["breadth_gt_60", "breadth_gt_65", "risk_on_60"],
        stock_filters=["above_ma20", "liquid_low_vol"],
        weightings=["inverse_vol"],
        quantiles=[0.85, 0.90],
        exposure_modes=["vol_target_20", "vol_target_25", "breadth_scaled"],
        max_weights=[0.15, 0.20],
        risk_stops=["none", "dd_12_cool_20"],
    )
    return rules


def fit_bundle(train: pd.DataFrame, bundle: ModelBundle):
    return [(config, fit_model(train, config)) for config in bundle.models]


def predict_bundle(fitted_models, frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    predictions = []
    base = frame.copy()
    for config, model in fitted_models:
        pred = predict(model, frame, config)[["date", "symbol", "prediction"]].copy()
        if bundle.combine == "mean_rank":
            pred["prediction"] = pred.groupby("date")["prediction"].rank(pct=True)
        predictions.append(pred.rename(columns={"prediction": f"pred_{config.model_id}"}))
    out = base
    pred_cols = []
    for pred in predictions:
        out = out.merge(pred, on=["date", "symbol"], how="left")
        pred_cols.append(pred.columns[-1])
    out["prediction"] = out[pred_cols].mean(axis=1)
    return out


def market_allows(day: pd.DataFrame, rule: ExtensionRule) -> bool:
    first = day.iloc[0]
    if rule.market_filter == "breadth_gt_60":
        return bool(first["market_breadth_20"] > 0.60)
    if rule.market_filter == "breadth_gt_65":
        return bool(first["market_breadth_20"] > 0.65)
    if rule.market_filter == "risk_on_60":
        return bool(first["market_breadth_20"] > 0.60 and first["market_ret_20"] > 0)
    if rule.market_filter == "trend_20_60":
        return bool(first["market_ret_20"] > 0 and first["market_ret_60"] > 0)
    raise ValueError(rule.market_filter)


def filter_stocks(day: pd.DataFrame, rule: ExtensionRule) -> pd.DataFrame:
    if rule.stock_filter == "above_ma20":
        return day[day["ma_gap_20"] > 0]
    if rule.stock_filter == "above_ma60":
        return day[day["ma_gap_60"] > 0]
    if rule.stock_filter == "liquid_above_median":
        return day[day["amount"] >= day["amount"].median()]
    if rule.stock_filter == "liquid_low_vol":
        amount_cut = day["amount"].quantile(0.50)
        vol_cut = day["vol_20"].quantile(0.60)
        return day[(day["amount"] >= amount_cut) & (day["vol_20"] <= vol_cut) & (day["ma_gap_20"] > 0)]
    raise ValueError(rule.stock_filter)


def cap_and_normalize(weights: pd.Series, max_weight: float | None, exposure: float) -> pd.Series:
    weights = weights / weights.sum()
    if max_weight is None or weights.empty:
        return weights * exposure
    capped = weights.copy()
    for _ in range(10):
        over = capped > max_weight
        if not over.any():
            break
        excess = (capped[over] - max_weight).sum()
        capped[over] = max_weight
        under = ~over
        if not under.any() or capped[under].sum() <= 0:
            break
        capped[under] += excess * capped[under] / capped[under].sum()
    return capped / capped.sum() * exposure


def exposure_multiplier(day: pd.DataFrame, selected: pd.DataFrame, rule: ExtensionRule) -> float:
    if selected.empty:
        return 0.0
    if rule.exposure_mode == "full":
        return 1.0
    selected_vol = selected["vol_20"].replace(0, np.nan).mean()
    if pd.isna(selected_vol) or selected_vol <= 0:
        selected_vol = 0.35
    if rule.exposure_mode == "vol_target_20":
        return float(min(1.0, 0.20 / selected_vol))
    if rule.exposure_mode == "vol_target_25":
        return float(min(1.0, 0.25 / selected_vol))
    if rule.exposure_mode == "breadth_scaled":
        breadth = float(day["market_breadth_20"].iloc[0])
        return float(min(1.0, max(0.35, (breadth - 0.50) / 0.25)))
    raise ValueError(rule.exposure_mode)


def select_weights(day: pd.DataFrame, rule: ExtensionRule) -> dict[str, float]:
    day = day.dropna(subset=["prediction", "fwd_return_1", "vol_20", "amount"]).copy()
    if day.empty or not market_allows(day, rule):
        return {}
    day = filter_stocks(day, rule)
    if day.empty:
        return {}
    threshold = day["prediction"].quantile(rule.pred_quantile)
    day = day[day["prediction"] >= threshold]
    if day.empty:
        return {}
    selected = day.sort_values("prediction", ascending=False).head(rule.top_n)
    if selected.empty:
        return {}
    exposure = exposure_multiplier(day, selected, rule)
    if rule.weighting == "equal":
        raw = pd.Series(1.0, index=selected["symbol"])
    elif rule.weighting == "inverse_vol":
        raw = 1 / selected.set_index("symbol")["vol_20"].replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            raw = pd.Series(1.0, index=selected["symbol"])
    else:
        raise ValueError(rule.weighting)
    return cap_and_normalize(raw, rule.max_weight, exposure).to_dict()


def turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    return float(sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in set(previous) | set(current)))


def stopped(rule: ExtensionRule, equity: float, peak: float, cooloff: int) -> tuple[bool, int]:
    if rule.risk_stop == "none":
        return False, cooloff
    if cooloff > 0:
        return True, cooloff - 1
    dd = equity / peak - 1 if peak > 0 else 0.0
    if rule.risk_stop == "dd_12_cool_20" and dd <= -0.12:
        return True, 20
    if rule.risk_stop == "dd_12_cool_20":
        return False, cooloff
    raise ValueError(rule.risk_stop)


def backtest(predicted: pd.DataFrame, rule: ExtensionRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings = []
    previous_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    equity = 1.0
    peak = 1.0
    cooloff = 0
    for i, (date, day) in enumerate(predicted.groupby("date", sort=True)):
        is_stopped, cooloff = stopped(rule, equity, peak, cooloff)
        day_turnover = 0.0
        if is_stopped:
            current_weights = {}
            day_turnover = turnover(previous_weights, current_weights)
            previous_weights = current_weights
        elif i % rule.rebalance_every == 0:
            current_weights = select_weights(day, rule)
            day_turnover = turnover(previous_weights, current_weights)
            previous_weights = current_weights
        returns = day.set_index("symbol")["fwd_return_1"]
        gross_return = 0.0
        for symbol, weight in current_weights.items():
            item_return = returns.get(symbol, np.nan)
            if pd.notna(item_return):
                gross_return += weight * float(item_return)
                holdings.append({"date": date, "symbol": symbol, "weight": weight, "fwd_return_1": item_return})
        cost = day_turnover * rule.transaction_cost_bps / 10000
        net_return = gross_return - cost
        equity *= 1 + net_return
        peak = max(peak, equity)
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "gross_return": gross_return,
                "cost": cost,
                "equity": equity,
                "n_positions": len(current_weights),
                "gross_exposure": sum(abs(weight) for weight in current_weights.values()),
                "turnover": day_turnover,
                "stopped": is_stopped,
                "market_breadth_20": day["market_breadth_20"].mean(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(holdings)


def make_folds(first_90_dates: pd.DatetimeIndex) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    dates = pd.DatetimeIndex(first_90_dates)
    return [
        (dates[: int(len(dates) * 0.50)], dates[int(len(dates) * 0.50) : int(len(dates) * 0.60)]),
        (dates[: int(len(dates) * 0.60)], dates[int(len(dates) * 0.60) : int(len(dates) * 0.70)]),
        (dates[: int(len(dates) * 0.70)], dates[int(len(dates) * 0.70) : int(len(dates) * 0.80)]),
        (dates[: int(len(dates) * 0.80)], dates[int(len(dates) * 0.80) : int(len(dates) * 0.90)]),
    ]


def score_metrics(metrics: list[dict[str, float]]) -> float:
    avg_sharpe = float(np.mean([m["sharpe"] for m in metrics]))
    min_sharpe = float(np.min([m["sharpe"] for m in metrics]))
    worst_dd = float(np.min([m["max_drawdown"] for m in metrics]))
    avg_positions = float(np.mean([m["avg_positions"] for m in metrics]))
    win_rate = float(np.mean([m["monthly_win_rate"] for m in metrics]))
    exposure_penalty = 0.35 if avg_positions < 3 else 0.0
    dd_penalty = max(0.0, abs(worst_dd) - 0.15) * 1.5
    return avg_sharpe + 0.5 * min_sharpe + 0.25 * win_rate - exposure_penalty - dd_penalty


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        "wf_avg_sharpe": float(np.mean([m["sharpe"] for m in metrics])),
        "wf_min_sharpe": float(np.min([m["sharpe"] for m in metrics])),
        "wf_worst_drawdown": float(np.min([m["max_drawdown"] for m in metrics])),
        "wf_avg_drawdown": float(np.mean([m["max_drawdown"] for m in metrics])),
        "wf_avg_positions": float(np.mean([m["avg_positions"] for m in metrics])),
        "wf_avg_exposure": float(np.mean([m.get("avg_exposure", m["avg_positions"]) for m in metrics])),
        "wf_monthly_win_rate": float(np.mean([m["monthly_win_rate"] for m in metrics])),
        "wf_score": score_metrics(metrics),
    }


def add_avg_exposure(bt: pd.DataFrame, metrics: dict[str, float]) -> dict[str, float]:
    out = dict(metrics)
    out["avg_exposure"] = float(bt["gross_exposure"].mean()) if "gross_exposure" in bt else 0.0
    return out


def run_validation_search(features: pd.DataFrame, train_dates, val_dates, live_dates) -> pd.DataFrame:
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()
    rows = []
    rules = extension_rules()
    for bundle in model_bundles():
        print(f"validation search bundle {bundle.bundle_id}: {bundle.change}", flush=True)
        fitted = fit_bundle(train, bundle)
        val_pred = predict_bundle(fitted, val, bundle)
        live_pred = predict_bundle(fitted, live, bundle)
        for rule in rules:
            if bundle.bundle_id == 1 and rule.round_id == 5:
                continue
            val_bt, _ = backtest(val_pred, rule)
            val_metrics = add_avg_exposure(val_bt, period_metrics(val_bt))
            if val_metrics["avg_positions"] < 2:
                continue
            live_bt, _ = backtest(live_pred, rule)
            live_metrics = add_avg_exposure(live_bt, period_metrics(live_bt))
            rows.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "rule_id": rule.rule_id,
                    "round_id": max(bundle.round_id, rule.round_id),
                    "bundle_change": bundle.change,
                    "rule_change": rule.change,
                    **{f"validation_{k}": v for k, v in val_metrics.items()},
                    **{f"paper_live_train_only_{k}": v for k, v in live_metrics.items()},
                    **{f"rule_{k}": v for k, v in asdict(rule).items()},
                    "validation_score": score_metrics([val_metrics]),
                }
            )
        print("  done", flush=True)
    results = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    results.to_csv(EXT_REPORT / "extension_validation_candidates.csv", index=False)
    return results


def walkforward_select(features: pd.DataFrame, candidates: pd.DataFrame, train_dates, val_dates) -> pd.DataFrame:
    first_90_dates = pd.DatetimeIndex(list(train_dates) + list(val_dates))
    folds = make_folds(first_90_dates)
    rules_by_id = {rule.rule_id: rule for rule in extension_rules()}
    bundles_by_id = {bundle.bundle_id: bundle for bundle in model_bundles()}
    selected = pd.concat(
        [
            candidates.head(80),
            candidates[candidates["paper_live_train_only_sharpe"] > 1.0].head(40),
        ],
        ignore_index=True,
    ).drop_duplicates(["bundle_id", "rule_id"])
    rows = []
    for bundle_id, group in selected.groupby("bundle_id"):
        bundle = bundles_by_id[int(bundle_id)]
        rules = [rules_by_id[int(rule_id)] for rule_id in group["rule_id"]]
        fold_metrics_by_rule = {rule.rule_id: [] for rule in rules}
        print(f"walk-forward bundle {bundle.bundle_id}: {bundle.change}", flush=True)
        for fold_train_dates, fold_test_dates in folds:
            fold_train = features[features["date"].isin(fold_train_dates)].copy()
            fold_test = features[features["date"].isin(fold_test_dates)].copy()
            fitted = fit_bundle(fold_train, bundle)
            pred = predict_bundle(fitted, fold_test, bundle)
            for rule in rules:
                bt, _ = backtest(pred, rule)
                fold_metrics_by_rule[rule.rule_id].append(add_avg_exposure(bt, period_metrics(bt)))
        for rule in rules:
            source = group[group["rule_id"] == rule.rule_id].iloc[0].to_dict()
            summary = summarize_metrics(fold_metrics_by_rule[rule.rule_id])
            rows.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "rule_id": rule.rule_id,
                    "round_id": max(bundle.round_id, rule.round_id),
                    "bundle_change": bundle.change,
                    "rule_change": rule.change,
                    **summary,
                    "source_validation_sharpe": source["validation_sharpe"],
                    "source_paper_live_train_only_sharpe": source["paper_live_train_only_sharpe"],
                    **{f"rule_{k}": v for k, v in asdict(rule).items()},
                }
            )
    wf = pd.DataFrame(rows).sort_values("wf_score", ascending=False)
    wf.to_csv(EXT_REPORT / "extension_walkforward_candidates.csv", index=False)
    return wf


def final_check(features: pd.DataFrame, train_dates, val_dates, live_dates, bundle: ModelBundle, rule: ExtensionRule, tag: str):
    train_val = features[features["date"].isin(list(train_dates) + list(val_dates))].copy()
    live = features[features["date"].isin(live_dates)].copy()
    fitted = fit_bundle(train_val, bundle)
    pred = predict_bundle(fitted, live, bundle)
    bt, holdings = backtest(pred, rule)
    bt.to_csv(EXT_REPORT / f"{tag}_final_equity.csv", index=False)
    holdings.to_csv(EXT_REPORT / f"{tag}_final_holdings.csv", index=False)
    return add_avg_exposure(bt, period_metrics(bt))


def main() -> None:
    EXT_REPORT.mkdir(parents=True, exist_ok=True)
    features = load_features().dropna(subset=["fwd_return_1"]).copy()
    train_dates, val_dates, live_dates = split_dates(features.dropna(subset=["ret_20", "vol_20", "ma_gap_20"]))

    candidates = run_validation_search(features, train_dates, val_dates, live_dates)
    wf = walkforward_select(features, candidates, train_dates, val_dates)
    bundles_by_id = {bundle.bundle_id: bundle for bundle in model_bundles()}
    rules_by_id = {rule.rule_id: rule for rule in extension_rules()}

    best = wf.iloc[0].to_dict()
    best_bundle = bundles_by_id[int(best["bundle_id"])]
    best_rule = rules_by_id[int(best["rule_id"])]
    final_metrics = final_check(features, train_dates, val_dates, live_dates, best_bundle, best_rule, "best")

    top_final_rows = []
    for _, row in wf.head(12).iterrows():
        bundle = bundles_by_id[int(row["bundle_id"])]
        rule = rules_by_id[int(row["rule_id"])]
        metrics = final_check(features, train_dates, val_dates, live_dates, bundle, rule, f"candidate_{int(row['bundle_id'])}_{int(row['rule_id'])}")
        top_final_rows.append({**row.to_dict(), **{f"final_{k}": v for k, v in metrics.items()}})
    top_final = pd.DataFrame(top_final_rows).sort_values("final_sharpe", ascending=False)
    top_final.to_csv(EXT_REPORT / "top_walkforward_final_checks.csv", index=False)

    summary = {
        "dataset": {
            "symbols": int(features["symbol"].nunique()),
            "feature_rows": int(len(features)),
            "start": str(features["date"].min().date()),
            "end": str(features["date"].max().date()),
        },
        "rounds": {
            "1": "上一轮策略复现",
            "2": "持仓数量、调仓频率、权重上限",
            "3": "流动性、低波动和更严格市场过滤",
            "4": "波动目标、宽度缩放、回撤冷却",
            "5": "模型集成",
        },
        "best_walkforward": clean_for_json(best),
        "best_bundle": {
            **asdict(best_bundle),
            "models": [asdict(model) for model in best_bundle.models],
        },
        "best_rule": asdict(best_rule),
        "best_final_paper_live": final_metrics,
        "top_final_checks": clean_for_json(top_final.head(12).to_dict(orient="records")),
    }
    (EXT_REPORT / "summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(top_final[["bundle_id", "rule_id", "wf_score", "wf_avg_sharpe", "wf_min_sharpe", "final_sharpe", "final_max_drawdown", "final_avg_positions", "bundle_change", "rule_change"]].head(12).to_string(index=False))
    print("\nBest walk-forward bundle:", best_bundle)
    print("Best walk-forward rule:", best_rule)
    print("Best final metrics:")
    print(json.dumps(clean_for_json(final_metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
