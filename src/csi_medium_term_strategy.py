from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from random_forest_a_share_strategy import PROCESSED_DIR, REPORT_DIR, clean_for_json, split_dates


REPORT_BASE = REPORT_DIR / "csi_medium_term"


FEATURES = (
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
    "trend_quality_20",
    "amount_ratio_5_20",
    "high_gap_20",
)


@dataclass(frozen=True)
class MediumModel:
    model_id: int
    round_id: int
    change: str
    target: str
    n_estimators: int = 160
    max_depth: int | None = 5
    min_samples_leaf: int = 120
    max_features: str | float = "sqrt"
    max_samples: float = 0.35
    random_state: int = 42


@dataclass(frozen=True)
class MediumRule:
    rule_id: int
    round_id: int
    change: str
    top_n: int
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_features(market: str) -> pd.DataFrame:
    paths = {
        "csi1000": PROCESSED_DIR / "csi1000_aggressive_features.parquet",
        "csi500": PROCESSED_DIR / "ashare_csi500_features.parquet",
    }
    if market == "combined":
        frames = []
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}")
            frame = pd.read_parquet(path)
            frame["source_index"] = name
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df = df.sort_values(["symbol", "date", "source_index"]).drop_duplicates(["symbol", "date"], keep="first")
    else:
        path = paths[market]
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.sort_values(["symbol", "date"]).copy()
    market_cols = ["market_ret_1", "market_ret_5", "market_ret_20", "market_ret_60", "market_vol_20", "market_breadth_20"]
    df = df.drop(columns=[col for col in market_cols if col in df.columns])
    market = (
        df.groupby("date", as_index=False)
        .agg(
            market_ret_1=("ret_1", "mean"),
            market_ret_5=("ret_5", "mean"),
            market_ret_20=("ret_20", "mean"),
            market_ret_60=("ret_60", "mean"),
            market_vol_20=("vol_20", "mean"),
            market_breadth_20=("ret_20", lambda x: float((x > 0).mean())),
        )
        .sort_values("date")
    )
    df = df.merge(market, on="date", how="left")
    if "trend_quality_20" not in df.columns:
        df["trend_quality_20"] = df["ret_20"] / df["vol_20"].replace(0, np.nan)
    if "amount_ratio_5_20" not in df.columns:
        pieces = []
        for _, group in df.groupby("symbol", sort=False):
            group = group.sort_values("date").copy()
            amount_5 = group["amount"].rolling(5).mean()
            amount_20 = group["amount"].rolling(20).mean()
            high_20 = group["high"].rolling(20).max()
            group["amount_ratio_5_20"] = amount_5 / amount_20.replace(0, np.nan)
            group["high_gap_20"] = group["close"] / high_20 - 1
            pieces.append(group)
        df = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])

    pieces = []
    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        group["fwd_return_20"] = group["close"].pct_change(20).shift(-20)
        group["fwd_return_40"] = group["close"].pct_change(40).shift(-40)
        pieces.append(group)
    df = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    df["target_rank_20"] = df.groupby("date")["fwd_return_20"].rank(pct=True)
    df["target_rank_40"] = df.groupby("date")["fwd_return_40"].rank(pct=True)
    df["target_excess_20"] = df["fwd_return_20"] - df.groupby("date")["fwd_return_20"].transform("mean")
    return df.replace([np.inf, -np.inf], np.nan)


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df[
        df["ret_1"].between(-0.22, 0.22)
        & df["fwd_return_1"].between(-0.22, 0.22)
        & (df["fwd_return_20"].between(-0.85, 2.5) | df["fwd_return_20"].isna())
        & (df["fwd_return_40"].between(-0.90, 3.0) | df["fwd_return_40"].isna())
    ].copy()
    print(f"[clean] kept {len(out)}/{before} rows", flush=True)
    return out


def target_col(target: str) -> str:
    return {
        "rank_20": "target_rank_20",
        "rank_40": "target_rank_40",
        "excess_20": "target_excess_20",
    }[target]


def models() -> list[MediumModel]:
    return [
        MediumModel(1, 1, "20日横截面排名：中线质量动量基线。", "rank_20"),
        MediumModel(2, 2, "40日横截面排名：更长持仓，降低短线噪声。", "rank_40", min_samples_leaf=150),
        MediumModel(3, 3, "20日超额收益：偏绝对收益和弹性。", "excess_20", min_samples_leaf=150),
    ]


def rules() -> list[MediumRule]:
    return [
        MediumRule(1, 1, "基线：Top20，20日调仓，8/12止损。", 20, 20, "trend_ok", "quality_trend", "inverse_vol", 0.80, 0.10, -0.08, -0.12, 60),
        MediumRule(2, 2, "更集中：Top10，20日调仓，严格8/10止损。", 10, 20, "trend_ok", "quality_trend", "inverse_vol", 0.85, 0.12, -0.08, -0.10, 50),
        MediumRule(3, 3, "更长期：Top20，40日调仓，10/15止损。", 20, 40, "trend_ok", "quality_trend", "inverse_vol", 0.80, 0.10, -0.10, -0.15, 90),
        MediumRule(4, 4, "低波质量：过滤高波动和过热，Top20。", 20, 20, "strong_trend", "quality_lowvol", "inverse_vol", 0.75, 0.08, -0.08, -0.12, 60),
        MediumRule(5, 5, "高相对强度：不追单日涨停，只要60日相对强。", 15, 20, "trend_ok", "relative_strength", "equal", 0.80, 0.10, -0.09, -0.13, 60),
        MediumRule(6, 6, "防守版：宽度差时降到现金，Top30。", 30, 20, "defensive", "quality_lowvol", "inverse_vol", 0.70, 0.06, -0.07, -0.10, 60),
        MediumRule(7, 7, "进攻但止损：Top8，10日检查，8/10止损。", 8, 10, "strong_trend", "relative_strength", "equal", 0.85, 0.15, -0.08, -0.10, 40),
    ]


def fit_model(train: pd.DataFrame, config: MediumModel) -> Pipeline:
    sample = train.dropna(subset=[target_col(config.target)]).copy()
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        max_samples=config.max_samples,
        bootstrap=True,
        random_state=config.random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    pipe.fit(sample[list(FEATURES)], sample[target_col(config.target)])
    return pipe


def predict(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[list(FEATURES)])
    return out


def market_allows(day: pd.DataFrame, rule: MediumRule) -> bool:
    first = day.iloc[0]
    breadth = float(first["market_breadth_20"])
    ret20 = float(first["market_ret_20"])
    ret60 = float(first["market_ret_60"])
    if rule.market_filter == "trend_ok":
        return breadth > 0.48 and (ret20 > -0.03 or ret60 > 0)
    if rule.market_filter == "strong_trend":
        return breadth > 0.55 and ret20 > 0 and ret60 > -0.02
    if rule.market_filter == "defensive":
        return breadth > 0.45 and ret60 > -0.06
    raise ValueError(rule.market_filter)


def tradable_mask(day: pd.DataFrame) -> pd.Series:
    symbols = day["symbol"].astype(str).str.zfill(6)
    wide = symbols.str.startswith(("300", "301", "688", "689"))
    limit = pd.Series(np.where(wide, 0.18, 0.092), index=day.index)
    return day["ret_1"] < limit


def filter_stocks(day: pd.DataFrame, rule: MediumRule) -> pd.DataFrame:
    day = day[tradable_mask(day)].copy()
    amount_cut = day["amount"].quantile(0.35)
    if rule.stock_filter == "quality_trend":
        return day[
            (day["amount"] >= amount_cut)
            & (day["ma_gap_60"] > -0.03)
            & (day["ret_60"] > 0)
            & (day["vol_20"] < day["vol_20"].quantile(0.80))
            & (day["rsi_14"].between(35, 78))
        ]
    if rule.stock_filter == "quality_lowvol":
        return day[
            (day["amount"] >= amount_cut)
            & (day["ma_gap_20"] > -0.02)
            & (day["ret_20"] > -0.03)
            & (day["vol_20"] < day["vol_20"].quantile(0.60))
            & (day["rsi_14"].between(35, 72))
        ]
    if rule.stock_filter == "relative_strength":
        return day[
            (day["amount"] >= amount_cut)
            & (day["rel_ret_60"] > day["rel_ret_60"].quantile(0.60))
            & (day["ret_20"] > -0.05)
            & (day["ret_1"] < 0.07)
        ]
    raise ValueError(rule.stock_filter)


def cap_weights(raw: pd.Series, max_weight: float) -> pd.Series:
    raw = raw / raw.sum()
    capped = raw.copy()
    for _ in range(12):
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


def select_weights(day: pd.DataFrame, rule: MediumRule) -> dict[str, float]:
    if not market_allows(day, rule):
        return {}
    day = day.dropna(subset=["prediction", "fwd_return_1", "vol_20", "amount", "close"]).copy()
    day = filter_stocks(day, rule)
    if day.empty:
        return {}
    threshold = day["prediction"].quantile(rule.pred_quantile)
    selected = day[day["prediction"] >= threshold].sort_values("prediction", ascending=False).head(rule.top_n)
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
    return cap_weights(raw, rule.max_weight).to_dict()


def turnover(old: dict[str, float], new: dict[str, float]) -> float:
    return float(sum(abs(old.get(s, 0.0) - new.get(s, 0.0)) for s in set(old) | set(new)))


def backtest(predicted: pd.DataFrame, rule: MediumRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings_rows = []
    weights: dict[str, float] = {}
    entry: dict[str, float] = {}
    peak: dict[str, float] = {}
    age: dict[str, int] = {}
    equity = 1.0

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
            target = select_weights(day, rule)
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
        for symbol, weight in weights.items():
            item_return = return_by_symbol.get(symbol, np.nan)
            if pd.notna(item_return):
                gross_return += weight * float(item_return)
                holdings_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "weight": weight,
                        "entry": entry.get(symbol),
                        "close": close_by_symbol.get(symbol),
                        "age": age.get(symbol, 0),
                        "fwd_return_1": item_return,
                    }
                )
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


def metrics(bt: pd.DataFrame) -> dict[str, float]:
    if bt.empty:
        return {"sharpe": 0.0, "annual_return": 0.0, "annual_vol": 0.0, "max_drawdown": 0.0}
    returns = bt["net_return"].fillna(0)
    std = returns.std(ddof=0)
    sharpe = 0.0 if std == 0 else float(returns.mean() / std * np.sqrt(252))
    annual_return = float(bt["equity"].iloc[-1] ** (252 / max(1, len(bt))) - 1)
    annual_vol = float(std * np.sqrt(252))
    drawdown = bt["equity"] / bt["equity"].cummax() - 1
    active = returns[returns != 0]
    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(bt["equity"].iloc[-1] - 1),
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "avg_positions": float(bt["n_positions"].mean()),
        "avg_exposure": float(bt["gross_exposure"].mean()),
        "avg_turnover": float(bt["turnover"].mean()),
        "stop_days": int((bt["stops"] > 0).sum()),
    }


def evaluate(features: pd.DataFrame, report_dir: Path):
    train_dates, val_dates, live_dates = split_dates(features.dropna(subset=["ret_60", "fwd_return_40"]))
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()
    rows = []
    for model_config in models():
        print(f"training model {model_config.model_id}: {model_config.change}", flush=True)
        model = fit_model(train, model_config)
        val_pred = predict(model, val)
        live_pred = predict(model, live)
        for rule in rules():
            if rule.round_id < model_config.round_id - 1:
                continue
            val_bt, _ = backtest(val_pred, rule)
            live_bt, _ = backtest(live_pred, rule)
            val_m = metrics(val_bt)
            live_m = metrics(live_bt)
            score = val_m["sharpe"] + 0.3 * val_m["win_rate"] - max(0.0, abs(val_m["max_drawdown"]) - 0.18)
            rows.append(
                {
                    "model_id": model_config.model_id,
                    "rule_id": rule.rule_id,
                    "round_id": max(model_config.round_id, rule.round_id),
                    "model_change": model_config.change,
                    "rule_change": rule.change,
                    "validation_score": score,
                    **{f"validation_{k}": v for k, v in val_m.items()},
                    **{f"paper_live_{k}": v for k, v in live_m.items()},
                    **{f"model_{k}": v for k, v in asdict(model_config).items()},
                    **{f"rule_{k}": v for k, v in asdict(rule).items()},
                }
            )
    candidates = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    candidates.to_csv(report_dir / "validation_candidates.csv", index=False)
    return candidates, train_dates, val_dates, live_dates


def final_check(features: pd.DataFrame, train_dates, val_dates, live_dates, model_config: MediumModel, rule: MediumRule, report_dir: Path, tag: str):
    train_val = features[features["date"].isin(list(train_dates) + list(val_dates))].copy()
    live = features[features["date"].isin(live_dates)].copy()
    model = fit_model(train_val, model_config)
    pred = predict(model, live)
    bt, holdings = backtest(pred, rule)
    bt.to_csv(report_dir / f"{tag}_equity.csv", index=False)
    holdings.to_csv(report_dir / f"{tag}_holdings.csv", index=False)
    return metrics(bt)


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / args.market
    ensure_dir(report_dir)
    features = clean_anomalies(load_features(args.market))
    candidates, train_dates, val_dates, live_dates = evaluate(features, report_dir)
    model_by_id = {m.model_id: m for m in models()}
    rule_by_id = {r.rule_id: r for r in rules()}
    rows = []
    for _, row in candidates.head(args.finalists).iterrows():
        model_config = model_by_id[int(row["model_id"])]
        rule = rule_by_id[int(row["rule_id"])]
        final_m = final_check(features, train_dates, val_dates, live_dates, model_config, rule, report_dir, f"candidate_{model_config.model_id}_{rule.rule_id}")
        rows.append({**row.to_dict(), **{f"final_{k}": v for k, v in final_m.items()}})
    top_final = pd.DataFrame(rows).sort_values("final_sharpe", ascending=False)
    top_final.to_csv(report_dir / "top_final_checks.csv", index=False)
    best = top_final.iloc[0].to_dict()
    best_model = model_by_id[int(best["model_id"])]
    best_rule = rule_by_id[int(best["rule_id"])]
    best_metrics = final_check(features, train_dates, val_dates, live_dates, best_model, best_rule, report_dir, "best")
    summary = {
        "objective": "Medium-term A-share stock selection with strict hard and trailing stops.",
        "market": args.market,
        "dataset": {
            "symbols": int(features["symbol"].nunique()),
            "rows": int(len(features)),
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
        "top_final_checks": clean_for_json(top_final.head(args.finalists).to_dict(orient="records")),
    }
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(top_final[["model_id", "rule_id", "validation_sharpe", "paper_live_sharpe", "final_sharpe", "final_annual_return", "final_max_drawdown", "final_avg_positions", "final_avg_exposure"]].head(args.finalists).to_string(index=False))
    print("\nBest model:", best_model)
    print("Best rule:", best_rule)
    print("Best final metrics:")
    print(json.dumps(clean_for_json(best_metrics), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["csi1000", "csi500", "combined"], default="csi1000")
    parser.add_argument("--finalists", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
