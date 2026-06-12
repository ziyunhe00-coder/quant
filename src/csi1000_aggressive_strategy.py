from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from adaptive_rf_strategy_search import enrich_features, period_metrics
from large_universe_fixed_strategy import fetch_yfinance_batch
from random_forest_a_share_strategy import PROCESSED_DIR, REPORT_DIR, clean_for_json, split_dates


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_DIR = ROOT / "data" / "universe"
CSI1000_RAW_DIR = ROOT / "data" / "raw_csi1000_yfinance"
CSI1000_REPORT_DIR = REPORT_DIR / "csi1000_aggressive"


AGGRESSIVE_FEATURES = (
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
    "market_ret_60",
    "market_vol_20",
    "market_breadth_20",
    "rel_ret_20",
    "rel_ret_60",
    "high_gap_20",
    "ret_accel_5_20",
    "amount_ratio_5_20",
    "price_volume_5",
    "trend_quality_20",
)


@dataclass(frozen=True)
class AggressiveModel:
    model_id: int
    round_id: int
    change: str
    target: str
    features: tuple[str, ...]
    n_estimators: int = 130
    max_depth: int | None = 5
    min_samples_leaf: int = 80
    max_features: str | float = "sqrt"
    max_samples: float = 0.30
    random_state: int = 42


@dataclass(frozen=True)
class AggressiveRule:
    rule_id: int
    round_id: int
    change: str
    top_n: int
    rebalance_every: int
    market_mode: str
    stock_filter: str
    weighting: str
    pred_quantile: float | None
    max_weight: float | None
    drawdown_stop: str
    transaction_cost_bps: float = 12.0


def ensure_dirs() -> None:
    for path in (UNIVERSE_DIR, CSI1000_RAW_DIR, CSI1000_REPORT_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def csi1000_universe(refresh: bool = False) -> pd.DataFrame:
    cache_path = UNIVERSE_DIR / "csi1000_constituents.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, dtype={"symbol": str})

    import akshare as ak

    df = ak.index_stock_cons(symbol="000852")
    out = df.rename(columns={"品种代码": "symbol", "品种名称": "name", "纳入日期": "in_date"})
    out = out[["symbol", "name", "in_date"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["yf_symbol"] = out["symbol"].map(lambda code: f"{code}.SS" if code.startswith("6") else f"{code}.SZ")
    out = out.drop_duplicates("symbol").sort_values("symbol")
    out.to_csv(cache_path, index=False)
    return out


def load_csi1000_prices(start_date: str, end_date: str, refresh: bool, max_symbols: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = csi1000_universe(refresh=refresh)
    if max_symbols:
        universe = universe.head(max_symbols)

    frames = []
    batch_size = 60
    for start in range(0, len(universe), batch_size):
        batch = universe.iloc[start : start + batch_size]
        frames.extend(fetch_yfinance_batch(batch, CSI1000_RAW_DIR, start_date, end_date, refresh, label="csi1000/yfinance"))
        print(f"[csi1000] loaded {len(frames)}/{min(start + batch_size, len(universe))} symbols", flush=True)
        time.sleep(0.4)
    if len(frames) < 500:
        raise RuntimeError(f"Only loaded {len(frames)} CSI1000 symbols; too few for aggressive test.")
    prices = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
    return prices, universe


def build_aggressive_features(prices: pd.DataFrame) -> pd.DataFrame:
    features = enrich_features(prices)
    pieces = []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        high_20 = group["high"].rolling(20).max()
        group["high_gap_20"] = group["close"] / high_20 - 1
        group["ret_accel_5_20"] = group["ret_5"] - group["ret_20"] / 4
        amount_5 = group["amount"].rolling(5).mean()
        amount_20 = group["amount"].rolling(20).mean()
        group["amount_ratio_5_20"] = amount_5 / amount_20.replace(0, np.nan)
        group["price_volume_5"] = group["ret_5"] * group["amount_z20"]
        group["trend_quality_20"] = group["ret_20"] / group["vol_20"].replace(0, np.nan)
        pieces.append(group)
    features = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    if "market_ret_60" not in features.columns:
        features = features.merge(features.groupby("date")["ret_60"].mean().rename("market_ret_60"), on="date", how="left")
    features["target_excess_5"] = features["fwd_return_5"] - features.groupby("date")["fwd_return_5"].transform("mean")
    return features.replace([np.inf, -np.inf], np.nan)


def clean_return_anomalies(features: pd.DataFrame, daily_limit: float) -> pd.DataFrame:
    before = len(features)
    cleaned = features.copy()
    mask = cleaned["ret_1"].between(-daily_limit, daily_limit) & cleaned["fwd_return_1"].between(
        -daily_limit, daily_limit
    )
    if "fwd_return_5" in cleaned.columns:
        mask &= cleaned["fwd_return_5"].between(-0.70, 1.50) | cleaned["fwd_return_5"].isna()
    cleaned = cleaned[mask].copy()
    cleaned["target_rank"] = cleaned.groupby("date")["fwd_return_1"].rank(pct=True)
    if "fwd_return_3" in cleaned.columns:
        cleaned["target_rank_3"] = cleaned.groupby("date")["fwd_return_3"].rank(pct=True)
    if "fwd_return_5" in cleaned.columns:
        cleaned["target_rank_5"] = cleaned.groupby("date")["fwd_return_5"].rank(pct=True)
        cleaned["target_excess_5"] = cleaned["fwd_return_5"] - cleaned.groupby("date")["fwd_return_5"].transform("mean")
    print(f"[clean] return anomaly filter kept {len(cleaned)}/{before} rows with daily_limit={daily_limit}", flush=True)
    return cleaned


def target_col(target: str) -> str:
    mapping = {
        "rank_1": "target_rank",
        "rank_5": "target_rank_5",
        "excess_5": "target_excess_5",
    }
    return mapping[target]


def models() -> list[AggressiveModel]:
    return [
        AggressiveModel(1, 1, "基准进攻模型：下一日横截面排名。", "rank_1", AGGRESSIVE_FEATURES),
        AggressiveModel(2, 4, "中期进攻模型：5日横截面排名。", "rank_5", AGGRESSIVE_FEATURES, min_samples_leaf=100),
        AggressiveModel(3, 4, "收益弹性模型：5日超额收益。", "excess_5", AGGRESSIVE_FEATURES, min_samples_leaf=100),
    ]


def rules(tradable_only: bool = False) -> list[AggressiveRule]:
    out = []
    rule_id = 1

    def add(round_id, change, top_ns, rebalances, market_modes, stock_filters, weightings, quantiles, max_weights, stops):
        nonlocal rule_id
        for top_n in top_ns:
            for rebalance in rebalances:
                for market_mode in market_modes:
                    for stock_filter in stock_filters:
                        for weighting in weightings:
                            for quantile in quantiles:
                                for max_weight in max_weights:
                                    for stop in stops:
                                        out.append(
                                            AggressiveRule(
                                                rule_id,
                                                round_id,
                                                change,
                                                top_n,
                                                rebalance,
                                                market_mode,
                                                stock_filter,
                                                weighting,
                                                quantile,
                                                max_weight,
                                                stop,
                                            )
                                        )
                                        rule_id += 1

    add(
        1,
        "满仓进攻基线：强势市场Top5/Top10。",
        [5, 10],
        [3, 5, 10],
        ["strong_full", "ladder"],
        ["above_ma20", "breakout_or_ma20"],
        ["equal", "inverse_vol"],
        [0.85, 0.90],
        [0.25, None],
        ["none"],
    )
    add(
        2,
        "更高暴露：弱市不完全空仓，目标平均暴露>=75%。",
        [5, 10],
        [3, 5],
        ["ladder_aggressive", "always_half_to_full"],
        ["above_ma20", "liquid_momentum"],
        ["equal", "inverse_vol"],
        [0.80, 0.85],
        [0.20, 0.25],
        ["none"],
    )
    add(
        3,
        "强趋势集中：只在高宽度/正趋势时Top5满仓。",
        [5, 8],
        [3, 5],
        ["very_strong_full", "strong_full"],
        ["breakout_or_ma20", "liquid_momentum"],
        ["equal", "inverse_vol"],
        [0.85, 0.90, 0.95],
        [0.25],
        ["none", "dd_30_cool_10"],
    )
    add(
        5,
        "回撤控制：保留高暴露但加入25/30%冷却。",
        [5, 10],
        [3, 5],
        ["ladder_aggressive", "always_half_to_full"],
        ["above_ma20", "liquid_momentum"],
        ["equal", "inverse_vol"],
        [0.80, 0.85],
        [0.20, 0.25],
        ["dd_25_cool_10", "dd_30_cool_10"],
    )
    add(
        6,
        "实盘可买性：过滤信号日接近涨停的股票，避免假设涨停价一定能成交。",
        [5, 8, 10],
        [3, 5],
        ["ladder_aggressive", "always_half_to_full"],
        ["tradable_above_ma20", "tradable_liquid_momentum"],
        ["equal", "inverse_vol"],
        [0.75, 0.80, 0.85],
        [0.20, 0.25],
        ["none", "dd_25_cool_10"],
    )
    if tradable_only:
        return [rule for rule in out if rule.stock_filter.startswith("tradable_")]
    return out


def fit_model(train: pd.DataFrame, config: AggressiveModel) -> Pipeline:
    sample = train.dropna(subset=[target_col(config.target)]).copy()
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        max_samples=config.max_samples,
        random_state=config.random_state,
        n_jobs=-1,
        bootstrap=True,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    pipe.fit(sample[list(config.features)], sample[target_col(config.target)])
    return pipe


def predict(model: Pipeline, frame: pd.DataFrame, config: AggressiveModel) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[list(config.features)])
    return out


def market_exposure(day: pd.DataFrame, rule: AggressiveRule) -> float:
    first = day.iloc[0]
    breadth = float(first["market_breadth_20"])
    ret5 = float(first["market_ret_5"])
    ret20 = float(first["market_ret_20"])
    ret60 = float(first["market_ret_60"])
    if rule.market_mode == "strong_full":
        return 1.0 if breadth > 0.55 and ret20 > 0 else 0.0
    if rule.market_mode == "very_strong_full":
        return 1.0 if breadth > 0.65 and ret20 > 0 and ret60 > 0 else 0.0
    if rule.market_mode == "ladder":
        if breadth > 0.60 and ret20 > 0:
            return 1.0
        if breadth > 0.50 or ret5 > 0:
            return 0.50
        return 0.0
    if rule.market_mode == "ladder_aggressive":
        if breadth > 0.58 and ret20 > 0:
            return 1.0
        if breadth > 0.48 or ret5 > 0:
            return 0.75
        return 0.30
    if rule.market_mode == "always_half_to_full":
        if breadth > 0.55 and ret20 > 0:
            return 1.0
        if breadth > 0.45 or ret5 > 0:
            return 0.70
        return 0.50
    raise ValueError(rule.market_mode)


def tradable_momentum_mask(day: pd.DataFrame) -> pd.Series:
    symbols = day["symbol"].astype(str).str.zfill(6)
    wide_limit = symbols.str.startswith(("300", "301", "688", "689"))
    limit_buffer = pd.Series(np.where(wide_limit, 0.18, 0.092), index=day.index)
    return day["ret_1"] < limit_buffer


def filter_stocks(day: pd.DataFrame, rule: AggressiveRule) -> pd.DataFrame:
    if rule.stock_filter == "above_ma20":
        return day[day["ma_gap_20"] > 0]
    if rule.stock_filter == "breakout_or_ma20":
        return day[(day["ma_gap_20"] > 0) | (day["high_gap_20"] > -0.03)]
    if rule.stock_filter == "liquid_momentum":
        amount_cut = day["amount"].quantile(0.45)
        return day[(day["amount"] >= amount_cut) & (day["ma_gap_20"] > 0) & (day["ret_5"] > -0.03)]
    if rule.stock_filter == "tradable_above_ma20":
        return day[(day["ma_gap_20"] > 0) & tradable_momentum_mask(day)]
    if rule.stock_filter == "tradable_liquid_momentum":
        amount_cut = day["amount"].quantile(0.45)
        return day[
            (day["amount"] >= amount_cut)
            & (day["ma_gap_20"] > 0)
            & (day["ret_5"] > -0.03)
            & tradable_momentum_mask(day)
        ]
    raise ValueError(rule.stock_filter)


def cap_weights(weights: pd.Series, max_weight: float | None) -> pd.Series:
    weights = weights / weights.sum()
    if max_weight is None:
        return weights
    capped = weights.copy()
    for _ in range(10):
        over = capped > max_weight
        if not over.any():
            break
        excess = (capped[over] - max_weight).sum()
        capped[over] = max_weight
        under = ~over
        if capped[under].sum() <= 0:
            break
        capped[under] += excess * capped[under] / capped[under].sum()
    return capped / capped.sum()


def select_weights(day: pd.DataFrame, rule: AggressiveRule) -> dict[str, float]:
    exposure = market_exposure(day, rule)
    if exposure <= 0:
        return {}
    day = day.dropna(subset=["prediction", "fwd_return_1", "vol_20", "amount"]).copy()
    day = filter_stocks(day, rule)
    if day.empty:
        return {}
    if rule.pred_quantile is not None:
        threshold = day["prediction"].quantile(rule.pred_quantile)
        day = day[day["prediction"] >= threshold]
    selected = day.sort_values("prediction", ascending=False).head(rule.top_n)
    if selected.empty:
        return {}
    if rule.weighting == "equal":
        raw = pd.Series(1.0, index=selected["symbol"])
    elif rule.weighting == "inverse_vol":
        raw = 1 / selected.set_index("symbol")["vol_20"].replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            raw = pd.Series(1.0, index=selected["symbol"])
    else:
        raise ValueError(rule.weighting)
    return (cap_weights(raw, rule.max_weight) * exposure).to_dict()


def turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    return float(sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in set(previous) | set(current)))


def risk_stopped(rule: AggressiveRule, equity: float, peak: float, cooloff: int) -> tuple[bool, int]:
    if rule.drawdown_stop == "none":
        return False, cooloff
    if cooloff > 0:
        return True, cooloff - 1
    dd = equity / peak - 1 if peak > 0 else 0.0
    if rule.drawdown_stop == "dd_25_cool_10" and dd <= -0.25:
        return True, 10
    if rule.drawdown_stop == "dd_30_cool_10" and dd <= -0.30:
        return True, 10
    return False, cooloff


def backtest(predicted: pd.DataFrame, rule: AggressiveRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings = []
    previous_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    equity = 1.0
    peak = 1.0
    cooloff = 0
    for i, (date, day) in enumerate(predicted.groupby("date", sort=True)):
        stopped, cooloff = risk_stopped(rule, equity, peak, cooloff)
        if stopped:
            new_weights = {}
            day_turnover = turnover(previous_weights, new_weights)
            current_weights = new_weights
            previous_weights = current_weights
        elif i % rule.rebalance_every == 0:
            current_weights = select_weights(day, rule)
            day_turnover = turnover(previous_weights, current_weights)
            previous_weights = current_weights
        else:
            day_turnover = 0.0

        returns = day.set_index("symbol")["fwd_return_1"]
        gross_return = 0.0
        for symbol, weight in current_weights.items():
            item_return = returns.get(symbol, np.nan)
            if pd.notna(item_return):
                gross_return += float(weight) * float(item_return)
                holdings.append({"date": date, "symbol": symbol, "weight": weight, "fwd_return_1": item_return})
        cost = day_turnover * rule.transaction_cost_bps / 10000
        net_return = gross_return - cost
        equity *= 1 + net_return
        peak = max(peak, equity)
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "equity": equity,
                "n_positions": len(current_weights),
                "gross_exposure": sum(abs(weight) for weight in current_weights.values()),
                "turnover": day_turnover,
                "stopped": stopped,
                "market_breadth_20": day["market_breadth_20"].mean(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(holdings)


def add_metrics(bt: pd.DataFrame) -> dict[str, float]:
    out = period_metrics(bt)
    out["avg_exposure"] = float(bt["gross_exposure"].mean()) if not bt.empty else 0.0
    out["calmar"] = 0.0 if out["max_drawdown"] == 0 else float(out["annual_return"] / abs(out["max_drawdown"]))
    return out


def score(metrics: dict[str, float]) -> float:
    exposure_penalty = max(0.0, 0.75 - metrics["avg_exposure"]) * 1.5
    dd_penalty = max(0.0, abs(metrics["max_drawdown"]) - 0.35) * 1.0
    return metrics["sharpe"] + 0.25 * metrics["calmar"] + 0.20 * metrics["monthly_win_rate"] - exposure_penalty - dd_penalty


def evaluate_model_rules(
    model_config: AggressiveModel,
    train: pd.DataFrame,
    val: pd.DataFrame,
    live: pd.DataFrame,
    candidate_rules: list[AggressiveRule],
):
    model = fit_model(train, model_config)
    val_pred = predict(model, val, model_config)
    live_pred = predict(model, live, model_config)
    rows = []
    artifacts = {}
    for rule in candidate_rules:
        if model_config.model_id != 1 and rule.round_id == 1:
            continue
        val_bt, val_h = backtest(val_pred, rule)
        val_m = add_metrics(val_bt)
        if val_m["avg_exposure"] < 0.50:
            continue
        live_bt, live_h = backtest(live_pred, rule)
        live_m = add_metrics(live_bt)
        row = {
            "model_id": model_config.model_id,
            "rule_id": rule.rule_id,
            "round_id": max(model_config.round_id, rule.round_id),
            "model_change": model_config.change,
            "rule_change": rule.change,
            **{f"validation_{k}": v for k, v in val_m.items()},
            **{f"paper_live_train_only_{k}": v for k, v in live_m.items()},
            **{f"model_{k}": v for k, v in asdict(model_config).items()},
            **{f"rule_{k}": v for k, v in asdict(rule).items()},
            "validation_score": score(val_m),
        }
        rows.append(row)
        artifacts[(model_config.model_id, rule.rule_id)] = (val_bt, val_h, live_bt, live_h)
    return rows, artifacts


def make_folds(first_90_dates: pd.DatetimeIndex):
    dates = pd.DatetimeIndex(first_90_dates)
    return [
        (dates[: int(len(dates) * 0.50)], dates[int(len(dates) * 0.50) : int(len(dates) * 0.60)]),
        (dates[: int(len(dates) * 0.60)], dates[int(len(dates) * 0.60) : int(len(dates) * 0.70)]),
        (dates[: int(len(dates) * 0.70)], dates[int(len(dates) * 0.70) : int(len(dates) * 0.80)]),
        (dates[: int(len(dates) * 0.80)], dates[int(len(dates) * 0.80) : int(len(dates) * 0.90)]),
    ]


def walkforward(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    train_dates,
    val_dates,
    candidate_rules: list[AggressiveRule],
) -> pd.DataFrame:
    selected = pd.concat(
        [
            candidates.sort_values("validation_score", ascending=False).head(80),
            candidates[candidates["paper_live_train_only_sharpe"] > 1.0].head(40),
        ],
        ignore_index=True,
    ).drop_duplicates(["model_id", "rule_id"])
    model_by_id = {model.model_id: model for model in models()}
    rule_by_id = {rule.rule_id: rule for rule in candidate_rules}
    folds = make_folds(pd.DatetimeIndex(list(train_dates) + list(val_dates)))
    rows = []
    for model_id, group in selected.groupby("model_id"):
        model_config = model_by_id[int(model_id)]
        group_rules = [rule_by_id[int(rule_id)] for rule_id in group["rule_id"]]
        fold_metrics = {rule.rule_id: [] for rule in group_rules}
        print(f"walk-forward model {model_id}: {model_config.change}", flush=True)
        for fold_train_dates, fold_test_dates in folds:
            fold_train = features[features["date"].isin(fold_train_dates)].copy()
            fold_test = features[features["date"].isin(fold_test_dates)].copy()
            model = fit_model(fold_train, model_config)
            pred = predict(model, fold_test, model_config)
            for rule in group_rules:
                bt, _ = backtest(pred, rule)
                fold_metrics[rule.rule_id].append(add_metrics(bt))
        for rule in group_rules:
            ms = fold_metrics[rule.rule_id]
            avg_sharpe = float(np.mean([m["sharpe"] for m in ms]))
            min_sharpe = float(np.min([m["sharpe"] for m in ms]))
            avg_exposure = float(np.mean([m["avg_exposure"] for m in ms]))
            avg_dd = float(np.mean([m["max_drawdown"] for m in ms]))
            wf_score = avg_sharpe + 0.45 * min_sharpe - max(0, 0.75 - avg_exposure) * 1.25
            source = group[group["rule_id"] == rule.rule_id].iloc[0].to_dict()
            rows.append(
                {
                    "model_id": model_config.model_id,
                    "rule_id": rule.rule_id,
                    "round_id": max(model_config.round_id, rule.round_id),
                    "wf_avg_sharpe": avg_sharpe,
                    "wf_min_sharpe": min_sharpe,
                    "wf_avg_exposure": avg_exposure,
                    "wf_avg_drawdown": avg_dd,
                    "wf_score": wf_score,
                    "source_validation_sharpe": source["validation_sharpe"],
                    "source_paper_live_train_only_sharpe": source["paper_live_train_only_sharpe"],
                    "model_change": model_config.change,
                    "rule_change": rule.change,
                    **{f"rule_{k}": v for k, v in asdict(rule).items()},
                }
            )
    wf = pd.DataFrame(rows).sort_values("wf_score", ascending=False)
    wf.to_csv(CSI1000_REPORT_DIR / "walkforward_candidates.csv", index=False)
    return wf


def final_check(features: pd.DataFrame, train_dates, val_dates, live_dates, model_config: AggressiveModel, rule: AggressiveRule, tag: str):
    train_val = features[features["date"].isin(list(train_dates) + list(val_dates))].copy()
    live = features[features["date"].isin(live_dates)].copy()
    model = fit_model(train_val, model_config)
    pred = predict(model, live, model_config)
    bt, holdings = backtest(pred, rule)
    bt.to_csv(CSI1000_REPORT_DIR / f"{tag}_final_equity.csv", index=False)
    holdings.to_csv(CSI1000_REPORT_DIR / f"{tag}_final_holdings.csv", index=False)
    return add_metrics(bt)


def run(args: argparse.Namespace) -> None:
    global CSI1000_REPORT_DIR
    CSI1000_REPORT_DIR = REPORT_DIR / args.report_name
    ensure_dirs()
    feature_path = PROCESSED_DIR / "csi1000_aggressive_features.parquet"
    if feature_path.exists() and not args.refresh_features:
        features = pd.read_parquet(feature_path)
        features["date"] = pd.to_datetime(features["date"])
    else:
        prices, universe = load_csi1000_prices(args.start_date, args.end_date, args.refresh_data, args.max_symbols)
        universe.to_csv(UNIVERSE_DIR / "csi1000_used.csv", index=False)
        features = build_aggressive_features(prices).dropna(subset=["fwd_return_1"]).copy()
        features.to_parquet(feature_path, index=False)

    if args.clean_return_limit:
        features = clean_return_anomalies(features, args.clean_return_limit)

    split_basis = features.dropna(subset=["ret_20", "vol_20", "ma_gap_20"])
    train_dates, val_dates, live_dates = split_dates(split_basis)
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()

    rows = []
    artifacts = {}
    candidate_rules = rules(tradable_only=args.tradable_only)
    for model_config in models():
        print(f"training model {model_config.model_id}: {model_config.change}", flush=True)
        model_rows, model_artifacts = evaluate_model_rules(model_config, train, val, live, candidate_rules)
        rows.extend(model_rows)
        artifacts.update(model_artifacts)
        print(f"  evaluated {len(model_rows)} candidates", flush=True)

    candidates = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    candidates.to_csv(CSI1000_REPORT_DIR / "validation_candidates.csv", index=False)
    wf = walkforward(features, candidates, train_dates, val_dates, candidate_rules)
    model_by_id = {model.model_id: model for model in models()}
    rule_by_id = {rule.rule_id: rule for rule in candidate_rules}
    top_final_rows = []
    for _, row in wf.head(12).iterrows():
        model_config = model_by_id[int(row["model_id"])]
        rule = rule_by_id[int(row["rule_id"])]
        final_m = final_check(features, train_dates, val_dates, live_dates, model_config, rule, f"candidate_{model_config.model_id}_{rule.rule_id}")
        top_final_rows.append({**row.to_dict(), **{f"final_{k}": v for k, v in final_m.items()}})
    top_final = pd.DataFrame(top_final_rows).sort_values("final_sharpe", ascending=False)
    top_final.to_csv(CSI1000_REPORT_DIR / "top_final_checks.csv", index=False)

    best = top_final.iloc[0].to_dict()
    best_model = model_by_id[int(best["model_id"])]
    best_rule = rule_by_id[int(best["rule_id"])]
    best_metrics = final_check(features, train_dates, val_dates, live_dates, best_model, best_rule, "best")
    summary = {
        "objective": "Small-cap aggressive strategy: high exposure, concentrated Top5/Top10, allow high drawdown while preserving Sharpe.",
        "tradable_only": bool(args.tradable_only),
        "dataset": {
            "symbols": int(features["symbol"].nunique()),
            "feature_rows": int(len(features)),
            "start": str(features["date"].min().date()),
            "end": str(features["date"].max().date()),
        },
        "split": {
            "train": f"{train_dates.min().date()} to {train_dates.max().date()} ({len(train_dates)} dates)",
            "validation": f"{val_dates.min().date()} to {val_dates.max().date()} ({len(val_dates)} dates)",
            "paper_live": f"{live_dates.min().date()} to {live_dates.max().date()} ({len(live_dates)} dates)",
        },
        "best_by_final_check": clean_for_json(best),
        "best_model": asdict(best_model),
        "best_rule": asdict(best_rule),
        "best_final_metrics": best_metrics,
        "top_final_checks": clean_for_json(top_final.head(12).to_dict(orient="records")),
    }
    (CSI1000_REPORT_DIR / "summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(top_final[["model_id", "rule_id", "wf_score", "wf_avg_sharpe", "wf_min_sharpe", "final_sharpe", "final_annual_return", "final_max_drawdown", "final_avg_exposure", "final_avg_positions"]].head(12).to_string(index=False))
    print("\nBest model:", best_model)
    print("Best rule:", best_rule)
    print("Best final metrics:")
    print(json.dumps(clean_for_json(best_metrics), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20180101")
    parser.add_argument("--end-date", default="20260610")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--clean-return-limit", type=float, default=0.22)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--tradable-only", action="store_true")
    parser.add_argument("--report-name", default="csi1000_aggressive")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
