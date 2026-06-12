from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from csi_fundamental_rf_small import REPORT_BASE, feature_columns, model_configs
from csi_fundamental_walkforward import combined_metrics, load_cached_features
from csi_medium_term_adaptive_walkforward import block_bootstrap, make_folds
from csi_medium_term_strategy import MediumRule, cap_weights, filter_stocks, market_allows, metrics, turnover
from random_forest_a_share_strategy import clean_for_json


@dataclass(frozen=True)
class ModelSpec:
    model_id: int
    feature_set: str
    change: str
    target: str = "target_rank_40"
    n_estimators: int = 260
    max_depth: int | None = 6
    min_samples_leaf: int = 60
    max_features: str | float = "sqrt"
    max_samples: float = 0.45
    random_state: int = 42


@dataclass(frozen=True)
class DiversifiedRule:
    rule_id: int
    change: str
    top_n: int
    min_positions: int
    rebalance_every: int
    market_filter: str
    stock_filter: str
    weighting: str
    pred_quantile: float
    max_weight: float
    hard_stop: float
    trailing_stop: float
    max_holding_days: int
    transaction_cost_bps: float = 14.0
    max_industry_weight: float = 1.0
    max_gross_exposure: float = 1.0
    industry_col: str = "industry_gate"


CHARACTERISTIC_COLS = [
    "val_log_total_mv_rank",
    "val_book_to_price_rank",
    "val_earnings_yield_rank",
    "val_sales_to_price_rank",
    "val_pb",
    "val_pe_ttm",
    "fin_roe",
    "fin_parent_profit",
    "fin_deduct_profit",
    "fin_ocf_per_share",
    "fin_debt_to_assets",
    "ret_60",
    "rel_ret_60",
    "vol_20",
]


def feature_group_map(features: pd.DataFrame) -> dict[str, list[str]]:
    value_cols = [col for col in features.columns if col.startswith("val_")]
    fin_cols = [col for col in features.columns if col.startswith("fin_")]
    market_cols = ["market_ret_20", "market_ret_60", "market_breadth_20"]
    price_cols = [
        "ret_20",
        "ret_60",
        "vol_20",
        "ma_gap_60",
        "rsi_14",
        "range_5",
        "rel_ret_20",
        "rel_ret_60",
    ]
    size_terms = ("total_mv", "float_mv", "shares", "log_total_mv", "log_float_mv")
    size_cols = [col for col in value_cols if any(term in col for term in size_terms)]
    valuation_cols = [col for col in value_cols if col not in size_cols]
    return {
        "size": size_cols,
        "valuation": valuation_cols,
        "financial": fin_cols,
        "market": [col for col in market_cols if col in features.columns],
        "price": [col for col in price_cols if col in features.columns],
    }


def model_specs(features: pd.DataFrame) -> list[ModelSpec]:
    groups = feature_group_map(features)
    specs = [
        ModelSpec(1, "value_fin_full", "估值+财报+市场状态，去掉大部分价量。", min_samples_leaf=50),
        ModelSpec(2, "value_fin_no_size", "估值+财报但移除市值/股本，检验是否只赚小盘。", min_samples_leaf=55),
        ModelSpec(3, "valuation_only", "只用估值和市值，不用财报。", min_samples_leaf=55),
        ModelSpec(4, "financial_only", "只用财报和市场状态，不用估值。", min_samples_leaf=55),
        ModelSpec(5, "price_value_fin", "价量+估值+财报全量组。", min_samples_leaf=70),
    ]
    # Drop empty feature-set variants in very sparse data, but keep normal order.
    return [spec for spec in specs if len(columns_for_spec(features, spec, groups)) >= 5]


def columns_for_spec(features: pd.DataFrame, spec: ModelSpec, groups: dict[str, list[str]] | None = None) -> list[str]:
    groups = groups or feature_group_map(features)
    if spec.feature_set == "value_fin_full":
        cols = groups["size"] + groups["valuation"] + groups["financial"] + groups["market"]
    elif spec.feature_set == "value_fin_no_size":
        cols = groups["valuation"] + groups["financial"] + groups["market"]
    elif spec.feature_set == "valuation_only":
        cols = groups["size"] + groups["valuation"] + groups["market"]
    elif spec.feature_set == "financial_only":
        cols = groups["financial"] + groups["market"]
    elif spec.feature_set == "price_value_fin":
        base = next(item for item in model_configs() if item.model_id == 3)
        cols = feature_columns(features, base)
    else:
        raise ValueError(spec.feature_set)
    return [col for col in dict.fromkeys(cols) if col in features.columns]


def diversified_rules(constrained_only: bool = False) -> list[DiversifiedRule]:
    base = [
        DiversifiedRule(
            301,
            "分散价值质量：Top12，至少10只，单票10%，20日调仓，12/18止损。",
            12,
            10,
            20,
            "trend_ok",
            "quality_trend",
            "inverse_vol",
            0.75,
            0.10,
            -0.12,
            -0.18,
            90,
        ),
        DiversifiedRule(
            302,
            "更分散：Top20，至少13只，单票8%，20日调仓，15/22止损。",
            20,
            13,
            20,
            "trend_ok",
            "quality_trend",
            "inverse_vol",
            0.70,
            0.08,
            -0.15,
            -0.22,
            120,
        ),
        DiversifiedRule(
            303,
            "长持仓防守：Top20，至少13只，单票8%，40日调仓，15/22止损。",
            20,
            13,
            40,
            "defensive",
            "quality_trend",
            "inverse_vol",
            0.70,
            0.08,
            -0.15,
            -0.22,
            140,
        ),
        DiversifiedRule(
            304,
            "相对强度确认：Top15，至少10只，单票10%，20日调仓，12/18止损。",
            15,
            10,
            20,
            "trend_ok",
            "relative_strength",
            "equal",
            0.72,
            0.10,
            -0.12,
            -0.18,
            100,
        ),
    ]
    constrained = [
        DiversifiedRule(
            401,
            "行业/暴露约束：Top15，至少10只，单票8%，行业30%，总暴露80%。",
            15,
            10,
            20,
            "trend_ok",
            "relative_strength",
            "equal",
            0.70,
            0.08,
            -0.12,
            -0.18,
            100,
            max_industry_weight=0.30,
            max_gross_exposure=0.80,
        ),
        DiversifiedRule(
            402,
            "行业/暴露约束：Top20，至少13只，单票8%，行业30%，总暴露85%。",
            20,
            13,
            20,
            "trend_ok",
            "quality_trend",
            "inverse_vol",
            0.68,
            0.08,
            -0.15,
            -0.22,
            120,
            max_industry_weight=0.30,
            max_gross_exposure=0.85,
        ),
        DiversifiedRule(
            403,
            "行业/暴露约束长持仓：Top20，至少13只，单票8%，行业30%，总暴露80%。",
            20,
            13,
            40,
            "defensive",
            "quality_trend",
            "inverse_vol",
            0.68,
            0.08,
            -0.15,
            -0.22,
            140,
            max_industry_weight=0.30,
            max_gross_exposure=0.80,
        ),
        DiversifiedRule(
            404,
            "更严格行业约束：Top20，至少15只，单票7%，行业25%，总暴露80%。",
            20,
            15,
            20,
            "trend_ok",
            "quality_trend",
            "inverse_vol",
            0.65,
            0.07,
            -0.15,
            -0.22,
            120,
            max_industry_weight=0.25,
            max_gross_exposure=0.80,
        ),
    ]
    return constrained if constrained_only else base + constrained


def as_medium_rule(rule: DiversifiedRule) -> MediumRule:
    return MediumRule(
        rule.rule_id,
        1,
        rule.change,
        rule.top_n,
        rule.rebalance_every,
        rule.market_filter,
        rule.stock_filter,
        rule.weighting,
        rule.pred_quantile,
        rule.max_weight,
        rule.hard_stop,
        rule.trailing_stop,
        rule.max_holding_days,
        rule.transaction_cost_bps,
    )


def load_industry_map(path: Path, industry_col: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", industry_col])
    industry = pd.read_csv(path, dtype={"symbol": str})
    industry["symbol"] = industry["symbol"].astype(str).str.zfill(6)
    if industry_col not in industry.columns:
        industry[industry_col] = industry.get("industry", "未知")
    keep = ["symbol", industry_col]
    if "industry" in industry.columns and "industry" not in keep:
        keep.append("industry")
    return industry[keep].drop_duplicates("symbol")


def attach_industry(features: pd.DataFrame, path: Path, industry_col: str) -> pd.DataFrame:
    industry = load_industry_map(path, industry_col)
    if industry.empty:
        features = features.copy()
        features[industry_col] = "未知"
        return features
    out = features.merge(industry, on="symbol", how="left")
    out[industry_col] = out[industry_col].fillna("未知")
    return out


def fit_model(train: pd.DataFrame, spec: ModelSpec, cols: list[str]) -> Pipeline:
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


def predict_frame(model: Pipeline, frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[cols])
    return out


def apply_industry_and_exposure_caps(weights: pd.Series, selected: pd.DataFrame, rule: DiversifiedRule) -> pd.Series:
    capped = weights.copy()
    if rule.max_industry_weight < 1.0 and rule.industry_col in selected.columns:
        industries = selected.set_index("symbol")[rule.industry_col].reindex(capped.index).fillna("未知")
        for _ in range(10):
            group_weight = capped.groupby(industries).sum()
            over = group_weight[group_weight > rule.max_industry_weight + 1e-12]
            if over.empty:
                break
            for industry, total in over.items():
                idx = industries[industries == industry].index
                capped.loc[idx] *= rule.max_industry_weight / total
    gross = capped.sum()
    if gross > rule.max_gross_exposure > 0:
        capped *= rule.max_gross_exposure / gross
    return capped[capped > 0]


def select_diversified_weights(day: pd.DataFrame, rule: DiversifiedRule) -> dict[str, float]:
    medium_rule = as_medium_rule(rule)
    if not market_allows(day, medium_rule):
        return {}
    day = day.dropna(subset=["prediction", "fwd_return_1", "vol_20", "amount", "close"]).copy()
    day = filter_stocks(day, medium_rule)
    if day.empty:
        return {}
    day = day.sort_values("prediction", ascending=False)
    threshold = day["prediction"].quantile(rule.pred_quantile)
    selected = day[day["prediction"] >= threshold].head(rule.top_n)
    needed = max(rule.min_positions, math.ceil(1.0 / rule.max_weight))
    if len(selected) < min(needed, len(day)):
        selected = day.head(min(rule.top_n, max(needed, rule.min_positions), len(day)))
    if selected.empty:
        return {}
    if rule.weighting == "equal":
        raw = pd.Series(1.0, index=selected["symbol"])
    elif rule.weighting == "inverse_vol":
        raw = 1 / selected.set_index("symbol")["vol_20"].replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        raise ValueError(rule.weighting)
    if raw.empty:
        return {}
    capped = cap_weights(raw, rule.max_weight)
    capped = apply_industry_and_exposure_caps(capped, selected, rule)
    return capped.to_dict()


def diversified_backtest(predicted: pd.DataFrame, rule: DiversifiedRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings_rows = []
    weights: dict[str, float] = {}
    entry: dict[str, float] = {}
    peak: dict[str, float] = {}
    age: dict[str, int] = {}
    equity = 1.0
    char_cols = [col for col in CHARACTERISTIC_COLS if col in predicted.columns]

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
            if pnl <= rule.hard_stop or dd_from_peak <= rule.trailing_stop or age[symbol] >= rule.max_holding_days:
                stopped.append(symbol)
        for symbol in stopped:
            weights.pop(symbol, None)
            entry.pop(symbol, None)
            peak.pop(symbol, None)
            age.pop(symbol, None)

        target = weights.copy()
        if i % rule.rebalance_every == 0:
            target = select_diversified_weights(day, rule)
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

        day_turnover = turnover(weights, target)
        weights = target
        gross_return = 0.0
        day_by_symbol = day.set_index("symbol")
        for symbol, weight in weights.items():
            item_return = return_by_symbol.get(symbol, np.nan)
            if pd.notna(item_return):
                gross_return += weight * float(item_return)
                row = {
                    "date": date,
                    "symbol": symbol,
                    "weight": weight,
                    "entry": entry.get(symbol),
                    "close": close_by_symbol.get(symbol),
                    "age": age.get(symbol, 0),
                    "prediction": day_by_symbol["prediction"].get(symbol, np.nan),
                    "fwd_return_1": item_return,
                    "contribution": weight * float(item_return),
                }
                if rule.industry_col in day_by_symbol.columns:
                    row[rule.industry_col] = day_by_symbol[rule.industry_col].get(symbol, "未知")
                for col in char_cols:
                    row[col] = day_by_symbol[col].get(symbol, np.nan)
                holdings_rows.append(row)
        cost = day_turnover * rule.transaction_cost_bps / 10000
        net_return = gross_return - cost
        equity *= 1 + net_return

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
                "turnover": day_turnover,
                "stops": len(stopped),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(holdings_rows)


def score_validation(m: dict[str, float]) -> float:
    if m.get("avg_positions", 0.0) < 1.0 or m.get("avg_exposure", 0.0) < 0.05:
        return -10.0
    position_penalty = max(0.0, 5.0 - m.get("avg_positions", 0.0)) * 0.22
    exposure_penalty = max(0.0, 0.35 - m.get("avg_exposure", 0.0)) * 2.0
    dd_penalty = max(0.0, abs(m.get("max_drawdown", 0.0)) - 0.22) * 1.4
    turnover_penalty = max(0.0, m.get("avg_turnover", 0.0) - 0.18) * 0.5
    return m["sharpe"] + 0.20 * m.get("win_rate", 0.0) - position_penalty - exposure_penalty - dd_penalty - turnover_penalty


def select_candidate(train: pd.DataFrame, val: pd.DataFrame, features: pd.DataFrame, constrained_only: bool) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    groups = feature_group_map(features)
    for spec in model_specs(features):
        cols = columns_for_spec(features, spec, groups)
        model = fit_model(train, spec, cols)
        val_pred = predict_frame(model, val, cols)
        for rule in diversified_rules(constrained_only):
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


def aggregate_feature_importance(rows: list[pd.DataFrame], features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    importance = pd.concat(rows, ignore_index=True)
    importance_summary = (
        importance.groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), selected_folds=("fold_id", "nunique"))
        .sort_values("mean_importance", ascending=False)
    )
    groups = feature_group_map(features)
    reverse = {}
    for group, cols in groups.items():
        for col in cols:
            reverse[col] = group
    importance["feature_group"] = importance["feature"].map(reverse).fillna("other")
    group_importance = (
        importance.groupby(["fold_id", "feature_group"], as_index=False)["importance"]
        .sum()
        .groupby("feature_group", as_index=False)
        .agg(mean_importance=("importance", "mean"), selected_folds=("fold_id", "nunique"))
        .sort_values("mean_importance", ascending=False)
    )
    return importance_summary, group_importance


def characteristic_exposure(features: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    cols = [col for col in CHARACTERISTIC_COLS if col in features.columns and col in holdings.columns]
    rows = []
    universe = features[features["date"].isin(pd.to_datetime(holdings["date"]).unique())]
    for col in cols:
        h = holdings[["date", "weight", col]].dropna().copy()
        if h.empty:
            continue
        held_daily = h.assign(weighted=h["weight"] * h[col]).groupby("date", as_index=False).agg(held_weighted_mean=("weighted", "sum"))
        uni_daily = universe.groupby("date", as_index=False).agg(universe_mean=(col, "mean"), universe_median=(col, "median"))
        merged = held_daily.merge(uni_daily, on="date", how="left")
        rows.append(
            {
                "feature": col,
                "held_weighted_mean": float(merged["held_weighted_mean"].mean()),
                "universe_mean": float(merged["universe_mean"].mean()),
                "universe_median": float(merged["universe_median"].mean()),
                "held_minus_universe_mean": float((merged["held_weighted_mean"] - merged["universe_mean"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("held_minus_universe_mean")


def contribution_by_symbol(holdings: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    names = features[["symbol", "name"]].drop_duplicates("symbol")
    out = (
        holdings.groupby("symbol", as_index=False)
        .agg(
            days_held=("date", "count"),
            avg_weight=("weight", "mean"),
            total_contribution=("contribution", "sum"),
            avg_prediction=("prediction", "mean"),
        )
        .merge(names, on="symbol", how="left")
        .sort_values("total_contribution", ascending=False)
    )
    return out


def contribution_by_industry(holdings: pd.DataFrame, industry_col: str) -> pd.DataFrame:
    if holdings.empty or industry_col not in holdings.columns:
        return pd.DataFrame()
    return (
        holdings.groupby(industry_col, as_index=False)
        .agg(
            days_held=("date", "count"),
            avg_weight=("weight", "mean"),
            total_contribution=("contribution", "sum"),
        )
        .sort_values("total_contribution", ascending=False)
    )


def summarize_equity(equity: pd.DataFrame, folds: pd.DataFrame, args: argparse.Namespace) -> dict:
    combined = combined_metrics(equity)
    returns = equity["net_return"].fillna(0) if not equity.empty else pd.Series(dtype=float)
    return {
        "combined": clean_for_json(combined),
        "folds": {
            "count": int(len(folds)),
            "avg_test_sharpe": float(folds["test_sharpe"].mean()) if len(folds) else 0.0,
            "median_test_sharpe": float(folds["test_sharpe"].median()) if len(folds) else 0.0,
            "min_test_sharpe": float(folds["test_sharpe"].min()) if len(folds) else 0.0,
            "max_test_sharpe": float(folds["test_sharpe"].max()) if len(folds) else 0.0,
            "positive_folds": int((folds["test_sharpe"] > 0).sum()) if len(folds) else 0,
        },
        "bootstrap_sharpe": block_bootstrap(returns, args.bootstrap, args.block_len, args.seed),
    }


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"improvement_n{args.max_symbols}_{args.report_name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    features = load_cached_features(args.max_symbols)
    features = attach_industry(features, Path(args.industry_map), args.industry_col)
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    if not folds:
        raise RuntimeError("No folds generated.")

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
        best, selection = select_candidate(train, val, features, args.constrained_only)
        selection["fold_id"] = fold_id
        selection_rows.append(selection)

        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        model = fit_model(train_full, best["model"], best["cols"])
        pred = predict_frame(model, test, best["cols"])
        bt, holdings = diversified_backtest(pred, best["rule"])
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
                "selected_feature_count": len(best["cols"]),
                "selected_rule_id": best["rule"].rule_id,
                "selected_rule_change": best["rule"].change,
                "validation_score": best["score"],
                "validation_sharpe": best["validation_sharpe"],
                **{f"test_{key}": value for key, value in test_m.items()},
            }
        )
        importances = model.named_steps["model"].feature_importances_
        importance_rows.append(
            pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "feature": best["cols"],
                    "importance": importances,
                    "feature_set": best["model"].feature_set,
                }
            )
        )

    folds_df = pd.DataFrame(fold_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    equity = pd.concat(equity_frames, ignore_index=True)
    holdings = pd.concat(holdings_frames, ignore_index=True) if holdings_frames else pd.DataFrame()
    equity["combined_equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    feature_importance, group_importance = aggregate_feature_importance(importance_rows, features)
    exposures = characteristic_exposure(features, holdings)
    symbol_contrib = contribution_by_symbol(holdings, features)
    industry_contrib = contribution_by_industry(holdings, args.industry_col)

    folds_df.to_csv(report_dir / "improved_walkforward_folds.csv", index=False)
    selections.to_csv(report_dir / "selection_candidates_by_fold.csv", index=False)
    equity.to_csv(report_dir / "improved_walkforward_equity.csv", index=False)
    holdings.to_csv(report_dir / "improved_walkforward_holdings.csv", index=False)
    feature_importance.to_csv(report_dir / "feature_importance.csv", index=False)
    group_importance.to_csv(report_dir / "feature_group_importance.csv", index=False)
    exposures.to_csv(report_dir / "holding_characteristic_exposure.csv", index=False)
    symbol_contrib.to_csv(report_dir / "symbol_contribution.csv", index=False)
    industry_contrib.to_csv(report_dir / "industry_contribution.csv", index=False)

    summary = {
        "objective": "Improve fundamental RF by enforcing diversification and explain return sources.",
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
        "constraints": {
            "constrained_only": args.constrained_only,
            "industry_map": args.industry_map,
            "industry_col": args.industry_col,
        },
        "walkforward": summarize_equity(equity, folds_df, args),
        "selected_feature_sets": clean_for_json(folds_df["selected_feature_set"].value_counts().to_dict()),
        "selected_rules": clean_for_json(folds_df["selected_rule_id"].value_counts().to_dict()),
        "top_feature_groups": clean_for_json(group_importance.head(10).to_dict(orient="records")),
        "top_features": clean_for_json(feature_importance.head(20).to_dict(orient="records")),
        "holding_exposures": clean_for_json(exposures.to_dict(orient="records")),
        "top_symbol_contributors": clean_for_json(symbol_contrib.head(15).to_dict(orient="records")),
        "bottom_symbol_contributors": clean_for_json(symbol_contrib.tail(15).to_dict(orient="records")),
        "industry_contributors": clean_for_json(industry_contrib.to_dict(orient="records")),
    }
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nImproved folds:")
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
    print("\nFeature groups:")
    print(group_importance.to_string(index=False))
    print("\nHolding exposures:")
    print(exposures.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(clean_for_json(summary["walkforward"]), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=120)
    parser.add_argument("--report-name", default="v1")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=194)
    parser.add_argument("--test-len", type=int, default=194)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--industry-map", default="data/processed/csi1000_industry_map_120.csv")
    parser.add_argument("--industry-col", default="industry_gate")
    parser.add_argument("--constrained-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
