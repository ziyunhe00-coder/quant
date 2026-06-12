from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from adaptive_rf_strategy_search import period_metrics
from random_forest_a_share_strategy import PROCESSED_DIR, REPORT_DIR, clean_for_json, split_dates


REPORT_DIR_CSI = REPORT_DIR / "csi500_iterations"
FEATURE_PATH = PROCESSED_DIR / "ashare_csi500_features.parquet"


@dataclass(frozen=True)
class ModelConfig:
    model_id: int
    iteration: int
    change: str
    target: str
    features: tuple[str, ...]
    n_estimators: int = 120
    max_depth: int | None = 5
    min_samples_leaf: int = 80
    max_features: str | float = "sqrt"
    max_samples: float = 0.30
    random_state: int = 42


@dataclass(frozen=True)
class RuleConfig:
    rule_id: int
    iteration: int
    change: str
    top_n: int
    rebalance_every: int
    market_filter: str
    stock_filter: str
    weighting: str
    pred_quantile: float | None = None
    exposure_mode: str = "full"
    risk_stop: str = "none"
    transaction_cost_bps: float = 12.0


CORE_FEATURES = (
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
)

DEFENSIVE_FEATURES = (
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
    "market_ret_60",
    "market_vol_20",
    "market_breadth_20",
)


def load_features() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURE_PATH}. Run large_universe_fixed_strategy.py first.")
    df = pd.read_parquet(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.sort_values(["symbol", "date"]).copy()
    if "market_ret_60" not in df.columns:
        market_ret_60 = df.groupby("date")["ret_60"].mean().rename("market_ret_60")
        df = df.merge(market_ret_60, on="date", how="left")
    if "target_excess_1" not in df.columns:
        df["target_excess_1"] = df["fwd_return_1"] - df.groupby("date")["fwd_return_1"].transform("mean")
        df["target_excess_5"] = df["fwd_return_5"] - df.groupby("date")["fwd_return_5"].transform("mean")
    return df.replace([np.inf, -np.inf], np.nan)


def target_col(target: str) -> str:
    mapping = {
        "return_1": "fwd_return_1",
        "return_5": "fwd_return_5",
        "rank_1": "target_rank",
        "rank_5": "target_rank_5",
        "excess_1": "target_excess_1",
        "excess_5": "target_excess_5",
    }
    return mapping[target]


def model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            model_id=1,
            iteration=1,
            change="基准模型：沿用上一轮浅树，预测下一日绝对收益。",
            target="return_1",
            features=DEFENSIVE_FEATURES,
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=80,
        ),
        ModelConfig(
            model_id=2,
            iteration=4,
            change="模型目标改为下一日超额收益，降低市场 beta 对预测的污染。",
            target="excess_1",
            features=CORE_FEATURES,
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=100,
        ),
        ModelConfig(
            model_id=3,
            iteration=4,
            change="模型目标改为下一日横截面排名，适配大股票池选股。",
            target="rank_1",
            features=CORE_FEATURES,
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=100,
        ),
        ModelConfig(
            model_id=4,
            iteration=4,
            change="模型目标改为5日超额收益，匹配低频持仓并降低日噪声。",
            target="excess_5",
            features=CORE_FEATURES,
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=100,
        ),
        ModelConfig(
            model_id=5,
            iteration=4,
            change="模型目标改为5日横截面排名，测试中期排序信号。",
            target="rank_5",
            features=CORE_FEATURES,
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=100,
        ),
    ]


def rule_configs() -> list[RuleConfig]:
    rules: list[RuleConfig] = []
    rule_id = 1

    def add(iteration, change, top_ns, rebalances, market_filters, stock_filters, weightings, quantiles, exposure_modes, risk_stops):
        nonlocal rule_id
        for top_n in top_ns:
            for rebalance_every in rebalances:
                for market_filter in market_filters:
                    for stock_filter in stock_filters:
                        for weighting in weightings:
                            for pred_quantile in quantiles:
                                for exposure_mode in exposure_modes:
                                    for risk_stop in risk_stops:
                                        rules.append(
                                            RuleConfig(
                                                rule_id=rule_id,
                                                iteration=iteration,
                                                change=change,
                                                top_n=top_n,
                                                rebalance_every=rebalance_every,
                                                market_filter=market_filter,
                                                stock_filter=stock_filter,
                                                weighting=weighting,
                                                pred_quantile=pred_quantile,
                                                exposure_mode=exposure_mode,
                                                risk_stop=risk_stop,
                                            )
                                        )
                                        rule_id += 1

    add(
        iteration=1,
        change="原逻辑复现：Top1、10日调仓、宽度>50%。",
        top_ns=[1],
        rebalances=[10],
        market_filters=["breadth_gt_50"],
        stock_filters=["none"],
        weightings=["equal"],
        quantiles=[None],
        exposure_modes=["full"],
        risk_stops=["none"],
    )
    add(
        iteration=2,
        change="组合层降风险：从单票改为Top5/10/20，加入等权和波动率倒数权重。",
        top_ns=[5, 10, 20],
        rebalances=[10, 20],
        market_filters=["breadth_gt_50"],
        stock_filters=["none", "above_ma20"],
        weightings=["equal", "inverse_vol"],
        quantiles=[None],
        exposure_modes=["full"],
        risk_stops=["none"],
    )
    add(
        iteration=3,
        change="市场状态过滤：提高宽度/趋势要求，避免弱市高波动暴露。",
        top_ns=[5, 10, 20],
        rebalances=[10, 20],
        market_filters=["breadth_gt_55", "breadth_gt_60", "risk_on_55", "trend_20_60"],
        stock_filters=["none", "above_ma20", "low_vol_half"],
        weightings=["inverse_vol"],
        quantiles=[None, 0.90],
        exposure_modes=["full"],
        risk_stops=["none"],
    )
    add(
        iteration=5,
        change="风控叠加：保留较优组合结构，加入波动目标和回撤冷却。",
        top_ns=[5, 10],
        rebalances=[10, 20],
        market_filters=["breadth_gt_60", "risk_on_55", "trend_20_60"],
        stock_filters=["above_ma20", "low_vol_half"],
        weightings=["inverse_vol"],
        quantiles=[None, 0.90],
        exposure_modes=["full", "vol_target_35"],
        risk_stops=["none", "dd_20_cool_20"],
    )
    return rules


def fit_model(train: pd.DataFrame, config: ModelConfig) -> Pipeline:
    col = target_col(config.target)
    sample = train.dropna(subset=[col]).copy()
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
    pipe.fit(sample[list(config.features)], sample[col])
    return pipe


def predict(model: Pipeline, frame: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[list(config.features)])
    return out


def market_allows(day: pd.DataFrame, rule: RuleConfig) -> bool:
    first = day.iloc[0]
    if rule.market_filter == "none":
        return True
    if rule.market_filter == "breadth_gt_50":
        return bool(first["market_breadth_20"] > 0.50)
    if rule.market_filter == "breadth_gt_55":
        return bool(first["market_breadth_20"] > 0.55)
    if rule.market_filter == "breadth_gt_60":
        return bool(first["market_breadth_20"] > 0.60)
    if rule.market_filter == "risk_on_50":
        return bool(first["market_breadth_20"] > 0.50 and first["market_ret_20"] > 0)
    if rule.market_filter == "risk_on_55":
        return bool(first["market_breadth_20"] > 0.55 and first["market_ret_20"] > 0)
    if rule.market_filter == "trend_20_60":
        return bool(first["market_ret_20"] > 0 and first["market_ret_60"] > 0)
    raise ValueError(rule.market_filter)


def filter_stocks(day: pd.DataFrame, rule: RuleConfig) -> pd.DataFrame:
    if rule.stock_filter == "none":
        return day
    if rule.stock_filter == "above_ma20":
        return day[day["ma_gap_20"] > 0]
    if rule.stock_filter == "above_ma60":
        return day[day["ma_gap_60"] > 0]
    if rule.stock_filter == "low_vol_half":
        threshold = day["vol_20"].quantile(0.50)
        return day[day["vol_20"] <= threshold]
    raise ValueError(rule.stock_filter)


def exposure_multiplier(day: pd.DataFrame, selected: pd.DataFrame, rule: RuleConfig) -> float:
    if selected.empty:
        return 0.0
    if rule.exposure_mode == "full":
        return 1.0
    if rule.exposure_mode == "half":
        return 0.5
    selected_vol = selected["vol_20"].replace(0, np.nan).mean()
    if pd.isna(selected_vol) or selected_vol <= 0:
        return 1.0
    if rule.exposure_mode == "vol_target_25":
        return float(min(1.0, 0.25 / selected_vol))
    if rule.exposure_mode == "vol_target_35":
        return float(min(1.0, 0.35 / selected_vol))
    raise ValueError(rule.exposure_mode)


def select_weights(day: pd.DataFrame, rule: RuleConfig) -> dict[str, float]:
    day = day.dropna(subset=["prediction", "fwd_return_1", "vol_20"]).copy()
    if day.empty or not market_allows(day, rule):
        return {}
    day = filter_stocks(day, rule)
    if day.empty:
        return {}
    if rule.pred_quantile is not None:
        threshold = day["prediction"].quantile(rule.pred_quantile)
        day = day[day["prediction"] >= threshold]
        if day.empty:
            return {}

    selected = day.sort_values("prediction", ascending=False).head(rule.top_n)
    if selected.empty:
        return {}
    exposure = exposure_multiplier(day, selected, rule)
    if exposure <= 0:
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
    weights = exposure * raw / raw.sum()
    return weights.to_dict()


def turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(previous) | set(current)
    return float(sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))


def risk_stop_state(rule: RuleConfig, equity: float, peak: float, cooloff: int) -> tuple[bool, int]:
    if rule.risk_stop == "none":
        return False, cooloff
    if cooloff > 0:
        return True, cooloff - 1
    dd = equity / peak - 1 if peak > 0 else 0
    if rule.risk_stop == "dd_15_cool_20" and dd <= -0.15:
        return True, 20
    if rule.risk_stop == "dd_20_cool_20" and dd <= -0.20:
        return True, 20
    return False, cooloff


def backtest(predicted: pd.DataFrame, rule: RuleConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings = []
    previous_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    equity = 1.0
    peak = 1.0
    cooloff = 0

    for i, (date, day) in enumerate(predicted.groupby("date", sort=True)):
        stopped, cooloff = risk_stop_state(rule, equity, peak, cooloff)
        rebalanced = i % rule.rebalance_every == 0
        day_turnover = 0.0
        if stopped:
            new_weights = {}
            day_turnover = turnover(previous_weights, new_weights)
            current_weights = new_weights
            previous_weights = current_weights
        elif rebalanced:
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
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "equity": equity,
                "n_positions": len(current_weights),
                "gross_exposure": sum(abs(w) for w in current_weights.values()),
                "turnover": day_turnover,
                "stopped": stopped,
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
        if len(sample) >= 20:
            values.append(sample["prediction"].corr(sample[col], method="spearman"))
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return {"rank_ic_mean": 0.0, "rank_ic_ir": 0.0}
    std = series.std(ddof=1)
    return {
        "rank_ic_mean": float(series.mean()),
        "rank_ic_ir": 0.0 if std == 0 or pd.isna(std) else float(series.mean() / std * math.sqrt(252)),
    }


def score(validation: dict[str, float]) -> float:
    dd_penalty = max(0.0, abs(validation["max_drawdown"]) - 0.20) * 2.0
    half_penalty = max(0.0, -validation["min_half_sharpe"]) * 0.75
    exposure_penalty = 0.4 if validation["avg_positions"] < 2 else 0.0
    return (
        validation["sharpe"]
        + 0.35 * validation["min_half_sharpe"]
        + 0.25 * validation["monthly_win_rate"]
        - dd_penalty
        - half_penalty
        - exposure_penalty
    )


def evaluate_model_rules(
    model_config: ModelConfig,
    rules: list[RuleConfig],
    train: pd.DataFrame,
    val: pd.DataFrame,
    live: pd.DataFrame,
) -> tuple[list[dict], dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
    model = fit_model(train, model_config)
    val_pred = predict(model, val, model_config)
    live_pred = predict(model, live, model_config)
    val_ic = rank_ic(val_pred, model_config.target)
    live_ic = rank_ic(live_pred, model_config.target)
    rows = []
    artifacts = {}
    for rule in rules:
        val_bt, val_holdings = backtest(val_pred, rule)
        val_metrics = period_metrics(val_bt)
        if val_metrics["avg_positions"] < 1:
            continue
        live_bt, live_holdings = backtest(live_pred, rule)
        live_metrics = period_metrics(live_bt)
        row = {
            "model_id": model_config.model_id,
            "rule_id": rule.rule_id,
            "iteration": max(model_config.iteration, rule.iteration),
            "model_change": model_config.change,
            "rule_change": rule.change,
            "target": model_config.target,
            "features": ",".join(model_config.features),
            **{f"validation_{key}": value for key, value in val_metrics.items()},
            **{f"paper_live_train_only_{key}": value for key, value in live_metrics.items()},
            **{f"validation_{key}": value for key, value in val_ic.items()},
            **{f"paper_live_train_only_{key}": value for key, value in live_ic.items()},
            **{f"model_{key}": value for key, value in asdict(model_config).items()},
            **{f"rule_{key}": value for key, value in asdict(rule).items()},
            "score": score(val_metrics),
        }
        rows.append(row)
        artifacts[rule.rule_id] = (val_bt, val_holdings, live_bt, live_holdings)
    return rows, artifacts


def final_retrain_check(
    features: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    live_dates: pd.DatetimeIndex,
    model_config: ModelConfig,
    rule: RuleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_val_dates = list(train_dates) + list(val_dates)
    train_val = features[features["date"].isin(train_val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()
    model = fit_model(train_val, model_config)
    live_pred = predict(model, live, model_config)
    bt, holdings = backtest(live_pred, rule)
    return bt, holdings, period_metrics(bt)


def main() -> None:
    REPORT_DIR_CSI.mkdir(parents=True, exist_ok=True)
    features = load_features().dropna(subset=["fwd_return_1"]).copy()
    train_dates, val_dates, live_dates = split_dates(features.dropna(subset=["ret_20", "vol_20", "ma_gap_20"]))
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()

    all_rules = rule_configs()
    all_models = model_configs()
    rows = []
    artifacts_by_key = {}
    for model_config in all_models:
        if model_config.model_id == 1:
            rules = all_rules
        else:
            rules = [rule for rule in all_rules if rule.iteration >= 3]
        print(f"training model {model_config.model_id}: {model_config.change}", flush=True)
        model_rows, artifacts = evaluate_model_rules(model_config, rules, train, val, live)
        rows.extend(model_rows)
        for rule_id, artifact in artifacts.items():
            artifacts_by_key[(model_config.model_id, rule_id)] = artifact
        print(f"  evaluated {len(model_rows)} candidates", flush=True)

    results = pd.DataFrame(rows).sort_values("score", ascending=False)
    results.to_csv(REPORT_DIR_CSI / "candidate_iterations.csv", index=False)
    if results.empty:
        raise RuntimeError("No candidates produced non-trivial exposure.")

    best = results.iloc[0].to_dict()
    best_model = next(model for model in all_models if model.model_id == int(best["model_id"]))
    best_rule = next(rule for rule in all_rules if rule.rule_id == int(best["rule_id"]))
    val_bt, val_holdings, live_bt, live_holdings = artifacts_by_key[(best_model.model_id, best_rule.rule_id)]
    val_bt.to_csv(REPORT_DIR_CSI / "best_validation_equity.csv", index=False)
    val_holdings.to_csv(REPORT_DIR_CSI / "best_validation_holdings.csv", index=False)
    live_bt.to_csv(REPORT_DIR_CSI / "best_paper_live_train_only_equity.csv", index=False)
    live_holdings.to_csv(REPORT_DIR_CSI / "best_paper_live_train_only_holdings.csv", index=False)

    final_bt, final_holdings, final_metrics = final_retrain_check(
        features, train_dates, val_dates, live_dates, best_model, best_rule
    )
    final_bt.to_csv(REPORT_DIR_CSI / "best_final_paper_live_equity.csv", index=False)
    final_holdings.to_csv(REPORT_DIR_CSI / "best_final_paper_live_holdings.csv", index=False)

    iteration_summary = (
        results.groupby("iteration")
        .agg(
            candidates=("score", "count"),
            best_score=("score", "max"),
            best_validation_sharpe=("validation_sharpe", "max"),
            median_validation_sharpe=("validation_sharpe", "median"),
        )
        .reset_index()
    )
    iteration_summary.to_csv(REPORT_DIR_CSI / "iteration_summary.csv", index=False)

    summary = {
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
        "selection_rule": "highest validation score = Sharpe + stability terms - drawdown/exposure penalties",
        "best_candidate": clean_for_json(best),
        "best_model": asdict(best_model),
        "best_rule": asdict(best_rule),
        "final_paper_live_retrained_on_90pct": final_metrics,
        "iteration_summary": clean_for_json(iteration_summary.to_dict(orient="records")),
    }
    (REPORT_DIR_CSI / "summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "iteration",
        "model_id",
        "rule_id",
        "validation_sharpe",
        "validation_max_drawdown",
        "validation_min_half_sharpe",
        "paper_live_train_only_sharpe",
        "paper_live_train_only_max_drawdown",
        "score",
        "model_change",
        "rule_change",
    ]
    print(results[display_cols].head(20).to_string(index=False))
    print("\nBest model:", best_model)
    print("Best rule:", best_rule)
    print("Final paper-live metrics after retraining on first 90%:")
    print(json.dumps(clean_for_json(final_metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
