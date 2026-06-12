from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from random_forest_a_share_strategy import (
    PROCESSED_DIR,
    RAW_DIR,
    REPORT_DIR,
    UNIVERSE,
    build_features,
    clean_for_json,
    ensure_dirs,
    load_market_data,
    metrics,
    split_dates,
)


ADAPTIVE_REPORT_DIR = REPORT_DIR / "adaptive_stage2"
BASE_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_10",
    "vol_20",
    "ma_gap_20",
    "ma_gap_60",
    "amount_z20",
    "rsi_14",
    "range_5",
    "vol_ratio_5_20",
    "market_ret_5",
    "market_ret_20",
    "market_vol_20",
    "market_breadth_20",
    "rel_ret_20",
    "rel_ret_60",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: int
    reason: str
    target: str
    features: tuple[str, ...]
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | float
    random_state: int = 42


@dataclass(frozen=True)
class PortfolioRule:
    rule_id: int
    reason: str
    direction: str
    top_n: int
    rebalance_every: int
    min_pred_quantile: float | None
    market_filter: str
    stock_filter: str
    weighting: str
    transaction_cost_bps: float = 12.0


def load_prices() -> pd.DataFrame:
    if not RAW_DIR.exists() or len(list(RAW_DIR.glob("*_qfq.csv"))) < 8:
        return load_market_data("20180101", "20260610", refresh=False)

    frames = []
    for symbol, name in UNIVERSE.items():
        matches = sorted(RAW_DIR.glob(f"{symbol}_*_qfq.csv"))
        if not matches:
            continue
        frame = pd.read_csv(matches[-1], parse_dates=["date"])
        frame["symbol"] = symbol
        frame["name"] = name
        frames.append(frame)
    if len(frames) < 8:
        return load_market_data("20180101", "20260610", refresh=False)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])


def enrich_features(prices: pd.DataFrame) -> pd.DataFrame:
    features = build_features(prices)
    pieces = []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        group["ret_10"] = group["close"].pct_change(10)
        group["ret_60"] = group["close"].pct_change(60)
        group["fwd_return_3"] = group["close"].pct_change(3).shift(-3)
        group["fwd_return_5"] = group["close"].pct_change(5).shift(-5)
        pieces.append(group)
    features = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])

    market = (
        features.groupby("date", as_index=False)
        .agg(
            market_ret_1=("ret_1", "mean"),
            market_ret_5=("ret_5", "mean"),
            market_ret_20=("ret_20", "mean"),
            market_vol_20=("vol_20", "mean"),
            market_breadth_20=("ret_20", lambda x: float((x > 0).mean())),
        )
        .sort_values("date")
    )
    features = features.merge(market, on="date", how="left")
    features["rel_ret_20"] = features["ret_20"] - features["market_ret_20"]
    features["rel_ret_60"] = features["ret_60"] - features.groupby("date")["ret_60"].transform("mean")
    features["target_rank_3"] = features.groupby("date")["fwd_return_3"].rank(pct=True)
    features["target_rank_5"] = features.groupby("date")["fwd_return_5"].rank(pct=True)
    return features.replace([np.inf, -np.inf], np.nan)


def target_col(target: str) -> str:
    mapping = {
        "return_1": "fwd_return_1",
        "return_3": "fwd_return_3",
        "return_5": "fwd_return_5",
        "rank_1": "target_rank",
        "rank_3": "target_rank_3",
        "rank_5": "target_rank_5",
    }
    return mapping[target]


def model_specs() -> list[ModelSpec]:
    core = tuple(BASE_FEATURES)
    compact = tuple(
        [
            "ret_1",
            "ret_3",
            "ret_5",
            "ret_20",
            "vol_20",
            "ma_gap_20",
            "ma_gap_60",
            "amount_z20",
            "rsi_14",
            "market_ret_20",
            "market_breadth_20",
            "rel_ret_20",
        ]
    )
    defensive = tuple(
        [
            "ret_1",
            "ret_5",
            "ret_20",
            "ret_60",
            "vol_10",
            "vol_20",
            "ma_gap_20",
            "ma_gap_60",
            "rsi_14",
            "range_5",
            "market_ret_5",
            "market_ret_20",
            "market_vol_20",
            "market_breadth_20",
        ]
    )
    specs = []
    model_id = 1
    for target, feature_set, leaf, depth, estimators, reason in [
        ("return_1", defensive, 25, 4, 350, "第二轮：第一轮最优来自浅树单日绝对收益，保留原结构。"),
        ("return_1", defensive, 35, 3, 450, "第二轮：进一步提高叶子样本数，降低过拟合。"),
        ("return_3", defensive, 25, 4, 450, "第二轮：同一浅树结构改为3日绝对收益，匹配低频持仓。"),
        ("return_5", defensive, 25, 4, 450, "第二轮：同一浅树结构改为5日绝对收益，检查持有期一致性。"),
    ]:
        specs.append(
            ModelSpec(
                model_id=model_id,
                reason=reason,
                target=target,
                features=feature_set,
                n_estimators=estimators,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features="sqrt",
            )
        )
        model_id += 1
    return specs


def fit_model(train: pd.DataFrame, spec: ModelSpec) -> Pipeline:
    col = target_col(spec.target)
    sample = train.dropna(subset=[col]).copy()
    model = RandomForestRegressor(
        n_estimators=spec.n_estimators,
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        max_features=spec.max_features,
        random_state=spec.random_state,
        n_jobs=-1,
        bootstrap=True,
    )
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    pipe.fit(sample[list(spec.features)], sample[col])
    return pipe


def predict(model: Pipeline, frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[list(spec.features)])
    return out


def candidate_rules() -> list[PortfolioRule]:
    rules = []
    rule_id = 1
    directions = ["highest"]
    top_ns = [1, 2, 3]
    rebalance_days = [5, 10, 15]
    quantiles = [None, 0.55, 0.65]
    market_filters = [
        "market_ret_5_positive",
        "market_ret_20_positive",
        "breadth_above_40",
        "breadth_above_50",
        "risk_on_loose",
        "risk_on",
    ]
    stock_filters = ["none", "stock_above_ma20"]
    weightings = ["equal", "inverse_vol"]

    for direction, top_n, rebalance_every, quantile, market_filter, stock_filter, weighting in product(
        directions,
        top_ns,
        rebalance_days,
        quantiles,
        market_filters,
        stock_filters,
        weightings,
    ):
        if direction == "lowest" and quantile is not None:
            continue
        if top_n == 1 and weighting == "inverse_vol":
            continue
        if rebalance_every == 1 and weighting == "inverse_vol":
            continue
        reason = (
            f"{direction}, Top{top_n}, {rebalance_every}日调仓, "
            f"预测阈值={quantile}, 市场过滤={market_filter}, 个股过滤={stock_filter}, 权重={weighting}"
        )
        rules.append(
            PortfolioRule(
                rule_id=rule_id,
                reason=reason,
                direction=direction,
                top_n=top_n,
                rebalance_every=rebalance_every,
                min_pred_quantile=quantile,
                market_filter=market_filter,
                stock_filter=stock_filter,
                weighting=weighting,
            )
        )
        rule_id += 1
    return rules


def market_allows(day: pd.DataFrame, rule: PortfolioRule) -> bool:
    first = day.iloc[0]
    if rule.market_filter == "none":
        return True
    if rule.market_filter == "market_ret_20_positive":
        return bool(first["market_ret_20"] > 0)
    if rule.market_filter == "market_ret_5_positive":
        return bool(first["market_ret_5"] > 0)
    if rule.market_filter == "breadth_above_40":
        return bool(first["market_breadth_20"] > 0.4)
    if rule.market_filter == "breadth_above_50":
        return bool(first["market_breadth_20"] > 0.5)
    if rule.market_filter == "risk_on_loose":
        return bool(first["market_ret_20"] > 0 or first["market_breadth_20"] > 0.5)
    if rule.market_filter == "risk_on":
        return bool(first["market_ret_20"] > 0 and first["market_breadth_20"] > 0.5)
    raise ValueError(rule.market_filter)


def filter_stocks(day: pd.DataFrame, rule: PortfolioRule) -> pd.DataFrame:
    if rule.stock_filter == "none":
        return day
    if rule.stock_filter == "stock_above_ma20":
        return day[day["ma_gap_20"] > 0]
    if rule.stock_filter == "stock_above_ma60":
        return day[day["ma_gap_60"] > 0]
    raise ValueError(rule.stock_filter)


def select_weights(day: pd.DataFrame, rule: PortfolioRule) -> dict[str, float]:
    day = day.dropna(subset=["prediction", "fwd_return_1", "vol_20"]).copy()
    if day.empty or not market_allows(day, rule):
        return {}
    day = filter_stocks(day, rule)
    if day.empty:
        return {}

    ascending = rule.direction == "lowest"
    if rule.min_pred_quantile is not None and rule.direction == "highest":
        threshold = day["prediction"].quantile(rule.min_pred_quantile)
        day = day[day["prediction"] >= threshold]
    day = day.sort_values("prediction", ascending=ascending).head(rule.top_n)
    if day.empty:
        return {}

    if rule.weighting == "equal":
        weight = 1 / len(day)
        return {symbol: weight for symbol in day["symbol"].tolist()}
    if rule.weighting == "inverse_vol":
        inv_vol = 1 / day.set_index("symbol")["vol_20"].replace(0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
        if inv_vol.empty:
            weight = 1 / len(day)
            return {symbol: weight for symbol in day["symbol"].tolist()}
        inv_vol = inv_vol / inv_vol.sum()
        return inv_vol.to_dict()
    raise ValueError(rule.weighting)


def turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(previous) | set(current)
    return sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)


def backtest(predicted: pd.DataFrame, rule: PortfolioRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings = []
    previous_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}

    for i, (date, day) in enumerate(predicted.groupby("date", sort=True)):
        rebalanced = i % rule.rebalance_every == 0
        day_turnover = 0.0
        if rebalanced:
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
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": gross_return - cost,
                "n_positions": len(current_weights),
                "turnover": day_turnover,
                "market_ret_1": day["market_ret_1"].mean(),
                "market_ret_20": day["market_ret_20"].mean(),
                "market_breadth_20": day["market_breadth_20"].mean(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(holdings)


def rank_ic(predicted: pd.DataFrame, target: str) -> dict[str, float]:
    col = target_col(target)
    values = []
    for _, day in predicted.groupby("date"):
        sample = day[["prediction", col]].dropna()
        if len(sample) >= 5:
            values.append(sample["prediction"].corr(sample[col], method="spearman"))
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return {"rank_ic_mean": 0.0, "rank_ic_std": 0.0, "rank_ic_ir": 0.0}
    std = series.std(ddof=1)
    return {
        "rank_ic_mean": float(series.mean()),
        "rank_ic_std": float(std) if pd.notna(std) else 0.0,
        "rank_ic_ir": 0.0 if std == 0 or pd.isna(std) else float(series.mean() / std * math.sqrt(252)),
    }


def period_metrics(bt: pd.DataFrame) -> dict[str, float]:
    bt = bt.copy()
    bt["date"] = pd.to_datetime(bt["date"])
    base = metrics(bt["net_return"])
    monthly = bt.set_index("date")["net_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    half = len(bt) // 2
    first = metrics(bt.iloc[:half]["net_return"]) if half > 10 else base
    second = metrics(bt.iloc[half:]["net_return"]) if len(bt) - half > 10 else base
    base.update(
        {
            "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0,
            "avg_positions": float(bt["n_positions"].mean()),
            "avg_turnover": float(bt["turnover"].mean()),
            "first_half_sharpe": first["sharpe"],
            "second_half_sharpe": second["sharpe"],
            "min_half_sharpe": min(first["sharpe"], second["sharpe"]),
        }
    )
    return base


def stability_pass(val: dict[str, float], live: dict[str, float]) -> bool:
    return (
        val["sharpe"] > 1.0
        and live["sharpe"] > 1.0
        and val["min_half_sharpe"] > 0.2
        and live["min_half_sharpe"] > 0.2
        and val["monthly_win_rate"] >= 0.5
        and live["monthly_win_rate"] >= 0.5
        and val["avg_positions"] >= 0.4
        and live["avg_positions"] >= 0.4
    )


def score_candidate(val: dict[str, float]) -> float:
    penalty = 0.0
    if val["avg_positions"] < 0.4:
        penalty += 2.0
    if val["monthly_win_rate"] < 0.5:
        penalty += 0.5
    if val["min_half_sharpe"] < 0:
        penalty += abs(val["min_half_sharpe"]) * 0.25
    return val["sharpe"] + 0.25 * val["min_half_sharpe"] + 0.5 * val["monthly_win_rate"] - penalty


def train_dates_for_final(train_dates, val_dates) -> list[pd.Timestamp]:
    return list(pd.DatetimeIndex(train_dates)) + list(pd.DatetimeIndex(val_dates))


def main() -> None:
    ensure_dirs()
    ADAPTIVE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    features = enrich_features(prices).dropna(subset=["fwd_return_1"]).copy()
    features.to_parquet(PROCESSED_DIR / "adaptive_features.parquet", index=False)

    train_dates, val_dates, live_dates = split_dates(features.dropna(subset=["ret_20", "vol_20", "ma_gap_20"]))
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()

    rows = []
    best = None
    best_artifacts = None
    rules = candidate_rules()
    for spec in model_specs():
        model = fit_model(train, spec)
        val_pred = predict(model, val, spec)
        live_pred = predict(model, live, spec)
        val_ic = rank_ic(val_pred, spec.target)
        live_ic = rank_ic(live_pred, spec.target)

        for rule in rules:
            val_bt, val_holdings = backtest(val_pred, rule)
            val_metrics = period_metrics(val_bt)
            if val_metrics["avg_positions"] < 0.25:
                continue
            live_bt, live_holdings = backtest(live_pred, rule)
            live_metrics = period_metrics(live_bt)
            row = {
                "model_id": spec.model_id,
                "rule_id": rule.rule_id,
                "model_reason": spec.reason,
                "rule_reason": rule.reason,
                "target": spec.target,
                "features": ",".join(spec.features),
                **{f"validation_{k}": v for k, v in val_metrics.items()},
                **{f"paper_live_{k}": v for k, v in live_metrics.items()},
                **{f"validation_{k}": v for k, v in val_ic.items()},
                **{f"paper_live_{k}": v for k, v in live_ic.items()},
                "score": score_candidate(val_metrics),
                "stable_pass": stability_pass(val_metrics, live_metrics),
                **{f"model_{k}": v for k, v in asdict(spec).items() if k not in {"features"}},
                **{f"rule_{k}": v for k, v in asdict(rule).items()},
            }
            rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
                best_artifacts = (spec, rule, val_bt, live_bt, val_holdings, live_holdings)

    results = pd.DataFrame(rows).sort_values(["stable_pass", "score"], ascending=[False, False])
    results.to_csv(ADAPTIVE_REPORT_DIR / "candidate_search.csv", index=False)

    if best_artifacts is None:
        raise RuntimeError("No candidate produced trades.")

    best_spec, best_rule, best_val_bt, best_live_bt, best_val_holdings, best_live_holdings = best_artifacts
    best_val_bt.to_csv(ADAPTIVE_REPORT_DIR / "best_validation_equity.csv", index=False)
    best_live_bt.to_csv(ADAPTIVE_REPORT_DIR / "best_paper_live_train_only_equity.csv", index=False)
    best_val_holdings.to_csv(ADAPTIVE_REPORT_DIR / "best_validation_holdings.csv", index=False)
    best_live_holdings.to_csv(ADAPTIVE_REPORT_DIR / "best_paper_live_train_only_holdings.csv", index=False)

    final_train = features[features["date"].isin(train_dates_for_final(train_dates, val_dates))].copy()
    final_live = features[features["date"].isin(live_dates)].copy()
    final_model = fit_model(final_train, best_spec)
    final_live_pred = predict(final_model, final_live, best_spec)
    final_bt, final_holdings = backtest(final_live_pred, best_rule)
    final_metrics = period_metrics(final_bt)
    final_bt.to_csv(ADAPTIVE_REPORT_DIR / "final_paper_live_equity.csv", index=False)
    final_holdings.to_csv(ADAPTIVE_REPORT_DIR / "final_paper_live_holdings.csv", index=False)
    final_live_pred.to_csv(ADAPTIVE_REPORT_DIR / "final_paper_live_predictions.csv", index=False)

    summary = {
        "stability_definition": (
            "validation Sharpe > 1, paper-live Sharpe > 1, both halves positive, "
            "monthly win rate >= 50%, and non-trivial position exposure"
        ),
        "split": {
            "train": f"{pd.DatetimeIndex(train_dates).min().date()} to {pd.DatetimeIndex(train_dates).max().date()}",
            "validation": f"{pd.DatetimeIndex(val_dates).min().date()} to {pd.DatetimeIndex(val_dates).max().date()}",
            "paper_live": f"{pd.DatetimeIndex(live_dates).min().date()} to {pd.DatetimeIndex(live_dates).max().date()}",
        },
        "searched_candidates": int(len(results)),
        "stable_candidates": int(results["stable_pass"].sum()),
        "best_by_validation_score": clean_for_json(best),
        "best_model": asdict(best_spec),
        "best_rule": asdict(best_rule),
        "final_paper_live_retrained_on_90pct": final_metrics,
    }
    (ADAPTIVE_REPORT_DIR / "summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "model_id",
        "rule_id",
        "validation_sharpe",
        "validation_min_half_sharpe",
        "validation_monthly_win_rate",
        "paper_live_sharpe",
        "paper_live_min_half_sharpe",
        "paper_live_monthly_win_rate",
        "stable_pass",
        "model_reason",
        "rule_reason",
    ]
    print(results[display_cols].head(20).to_string(index=False))
    print("\nBest model:", best_spec)
    print("Best rule:", best_rule)
    print("Final paper-live metrics after retraining on first 90%:")
    print(json.dumps(clean_for_json(final_metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
