from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from random_forest_a_share_strategy import PROCESSED_DIR, REPORT_DIR, clean_for_json


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_DIR = ROOT / "data" / "universe"
INTRADAY_RAW_DIR = ROOT / "data" / "raw_csi1000_intraday_15m"
INTRADAY_REPORT_DIR = REPORT_DIR / "csi1000_intraday_momentum"


FEATURES = (
    "ret_prev_close",
    "ret_from_open",
    "bar_ret_1",
    "bar_ret_2",
    "bar_ret_4",
    "above_vwap",
    "range_from_open",
    "cum_amount_ratio_10",
    "amount_ratio_10",
    "slot",
    "market_ret_prev_close",
    "market_ret_from_open",
    "market_breadth_open",
    "market_breadth_prev_close",
)


@dataclass(frozen=True)
class IntradayModel:
    model_id: int
    round_id: int
    change: str
    target: str
    n_estimators: int = 120
    max_depth: int | None = 5
    min_samples_leaf: int = 40
    max_features: str | float = "sqrt"
    max_samples: float = 0.45
    random_state: int = 42


@dataclass(frozen=True)
class IntradayRule:
    rule_id: int
    round_id: int
    change: str
    top_n: int
    signal_start: str
    signal_end: str
    min_ret_prev_close: float
    max_ret_prev_close_main: float
    max_ret_prev_close_wide: float
    min_ret_from_open: float
    max_ret_from_open: float
    min_above_vwap: float
    min_cum_amount_ratio: float
    min_market_breadth: float
    min_prediction_quantile: float
    transaction_cost_bps: float = 24.0


def ensure_dirs() -> None:
    for path in (INTRADAY_RAW_DIR, INTRADAY_REPORT_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def select_liquid_symbols(max_symbols: int) -> pd.DataFrame:
    used_path = UNIVERSE_DIR / "csi1000_used.csv"
    if not used_path.exists():
        raise RuntimeError("Missing csi1000_used.csv. Run the daily CSI1000 script first.")
    universe = pd.read_csv(used_path, dtype={"symbol": str})
    universe["symbol"] = universe["symbol"].str.zfill(6)

    feature_path = PROCESSED_DIR / "csi1000_aggressive_features.parquet"
    if not feature_path.exists():
        return universe.head(max_symbols)
    features = pd.read_parquet(feature_path, columns=["date", "symbol", "amount"])
    features["date"] = pd.to_datetime(features["date"])
    features["symbol"] = features["symbol"].astype(str).str.zfill(6)
    cutoff = features["date"].max() - pd.Timedelta(days=90)
    liquidity = (
        features[features["date"] >= cutoff]
        .groupby("symbol", as_index=False)["amount"]
        .median()
        .rename(columns={"amount": "median_amount"})
        .sort_values("median_amount", ascending=False)
    )
    out = liquidity.merge(universe, on="symbol", how="left").head(max_symbols)
    return out[["symbol", "name", "median_amount"]].copy()


def normalize_intraday(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename = {
        "时间": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    out = raw.rename(columns=rename)
    needed = ["datetime", "open", "close", "high", "low", "volume", "amount"]
    if any(col not in out.columns for col in needed):
        return pd.DataFrame()
    out = out[needed].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["datetime", "open", "close", "high", "low"])
    out["symbol"] = symbol
    out["date"] = out["datetime"].dt.date.astype(str)
    out["time"] = out["datetime"].dt.strftime("%H:%M")
    return out.sort_values("datetime")


def fetch_one_intraday(symbol: str, start_date: str, end_date: str, period: str, refresh: bool) -> pd.DataFrame:
    cache_path = INTRADAY_RAW_DIR / f"{symbol}_{period}_qfq.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["datetime"])

    import akshare as ak

    start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 09:30:00"
    end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]} 15:00:00"
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_hist_min_em(symbol=symbol, start_date=start, end_date=end, period=period, adjust="qfq")
            out = normalize_intraday(raw, symbol)
            if not out.empty:
                out.to_csv(cache_path, index=False)
            return out
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.2 + attempt)
    print(f"[minute] failed {symbol}: {last_exc}", flush=True)
    return pd.DataFrame()


def load_intraday(symbols: pd.DataFrame, start_date: str, end_date: str, period: str, refresh: bool) -> pd.DataFrame:
    frames = []
    for i, row in enumerate(symbols.itertuples(index=False), start=1):
        frame = fetch_one_intraday(row.symbol, start_date, end_date, period, refresh)
        if len(frame) >= 120:
            frames.append(frame)
        print(f"[minute] loaded {len(frames)}/{i}: {row.symbol}", flush=True)
        time.sleep(0.25)
    if len(frames) < 20:
        raise RuntimeError(f"Only loaded {len(frames)} symbols; too few for intraday test.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "datetime"])


def is_wide_limit_symbol(symbols: pd.Series) -> pd.Series:
    return symbols.astype(str).str.zfill(6).str.startswith(("300", "301", "688", "689"))


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.sort_values(["symbol", "datetime"]).copy()
    pieces = []
    for _, group in bars.groupby("symbol", sort=False):
        group = group.sort_values("datetime").copy()
        group["slot"] = group.groupby("date").cumcount()
        day_open = group.groupby("date")["open"].transform("first")
        day_high = group.groupby("date")["high"].cummax()
        day_low = group.groupby("date")["low"].cummin()
        day_close = group.groupby("date")["close"].transform("last")
        prev_close_by_day = group.groupby("date")["close"].last().shift(1)
        group = group.merge(prev_close_by_day.rename("prev_close"), left_on="date", right_index=True, how="left")
        group["ret_prev_close"] = group["close"] / group["prev_close"] - 1
        group["ret_from_open"] = group["close"] / day_open - 1
        group["bar_ret_1"] = group["close"].pct_change(1)
        group["bar_ret_2"] = group["close"].pct_change(2)
        group["bar_ret_4"] = group["close"].pct_change(4)
        group["cum_amount"] = group.groupby("date")["amount"].cumsum()
        group["cum_volume"] = group.groupby("date")["volume"].cumsum()
        group["vwap"] = group["cum_amount"] / (group["cum_volume"].replace(0, np.nan) * 100)
        group["above_vwap"] = group["close"] / group["vwap"] - 1
        group["range_from_open"] = (day_high - day_low) / day_open
        group["exit_next_close"] = group.groupby("date")["close"].last().shift(-1).reindex(group["date"]).to_numpy()
        group["entry_next_open"] = group["open"].shift(-1)
        group["entry_next_datetime"] = group["datetime"].shift(-1)
        group.loc[group["date"] != group["date"].shift(-1), "entry_next_open"] = np.nan

        daily_slot = group[["date", "slot", "cum_amount", "amount"]].copy()
        avg_cum = (
            daily_slot.pivot(index="date", columns="slot", values="cum_amount")
            .rolling(10, min_periods=3)
            .mean()
            .shift(1)
            .stack()
            .rename("avg_cum_amount_10")
            .reset_index()
        )
        avg_bar = (
            daily_slot.pivot(index="date", columns="slot", values="amount")
            .rolling(10, min_periods=3)
            .mean()
            .shift(1)
            .stack()
            .rename("avg_amount_10")
            .reset_index()
        )
        group = group.merge(avg_cum, on=["date", "slot"], how="left").merge(avg_bar, on=["date", "slot"], how="left")
        group["cum_amount_ratio_10"] = group["cum_amount"] / group["avg_cum_amount_10"].replace(0, np.nan)
        group["amount_ratio_10"] = group["amount"] / group["avg_amount_10"].replace(0, np.nan)
        group["target_return"] = group["exit_next_close"] / group["entry_next_open"] - 1
        pieces.append(group)

    features = pd.concat(pieces, ignore_index=True)
    market = (
        features.groupby("datetime", as_index=False)
        .agg(
            market_ret_prev_close=("ret_prev_close", "mean"),
            market_ret_from_open=("ret_from_open", "mean"),
            market_breadth_open=("ret_from_open", lambda x: float((x > 0).mean())),
            market_breadth_prev_close=("ret_prev_close", lambda x: float((x > 0).mean())),
        )
        .sort_values("datetime")
    )
    features = features.merge(market, on="datetime", how="left")
    features["target_rank"] = features.groupby("datetime")["target_return"].rank(pct=True)
    return features.replace([np.inf, -np.inf], np.nan)


def models() -> list[IntradayModel]:
    return [
        IntradayModel(1, 1, "基线：预测下一交易日收益。", "target_return", min_samples_leaf=35),
        IntradayModel(2, 2, "横截面排名：降低极端收益点影响。", "target_rank", min_samples_leaf=45),
        IntradayModel(3, 4, "更浅模型：减少短样本过拟合。", "target_rank", max_depth=4, min_samples_leaf=60, max_samples=0.55),
    ]


def rules() -> list[IntradayRule]:
    return [
        IntradayRule(1, 1, "基线：早盘强势未涨停，放量站上VWAP。", 3, "10:00", "11:15", 0.025, 0.085, 0.165, 0.006, 9.0, 0.002, 1.25, 0.52, 0.80),
        IntradayRule(2, 2, "更保守：避开过热，要求更强市场宽度。", 3, "10:00", "11:15", 0.020, 0.070, 0.145, 0.004, 9.0, 0.001, 1.15, 0.56, 0.80),
        IntradayRule(3, 3, "午盘确认：早盘强势后仍在VWAP上方。", 3, "11:00", "14:00", 0.018, 0.075, 0.150, 0.006, 9.0, 0.001, 1.05, 0.54, 0.80),
        IntradayRule(4, 4, "更分散：Top5，降低单票运气。", 5, "10:00", "14:15", 0.015, 0.075, 0.150, 0.003, 9.0, 0.000, 1.00, 0.52, 0.75),
        IntradayRule(5, 5, "高质量强势：更高放量和预测分位。", 3, "10:15", "13:45", 0.025, 0.080, 0.160, 0.008, 9.0, 0.003, 1.40, 0.56, 0.85),
        IntradayRule(6, 6, "覆盖率修正：宽松强势，要求可交易但不强求站稳VWAP。", 5, "10:00", "14:15", 0.005, 0.090, 0.170, -0.005, 9.0, -0.003, 0.70, 0.40, 0.65),
        IntradayRule(7, 7, "温和动量：稍强于开盘和市场，优先增加交易天数。", 5, "10:00", "14:15", 0.010, 0.080, 0.160, 0.000, 9.0, -0.001, 0.80, 0.45, 0.70),
        IntradayRule(8, 8, "盘中回踩：允许开盘后小回落，但维持日内强势。", 5, "10:30", "14:30", 0.010, 0.090, 0.170, -0.005, 9.0, -0.004, 0.70, 0.45, 0.60),
        IntradayRule(9, 9, "更分散覆盖：Top8，降低单日单票噪声。", 8, "10:00", "14:30", 0.000, 0.095, 0.175, -0.008, 9.0, -0.004, 0.60, 0.38, 0.60),
        IntradayRule(10, 10, "低吸回踩：不追涨，只买平盘附近转强。", 5, "10:00", "14:15", -0.020, 0.030, 0.050, -0.015, 0.010, -0.003, 0.70, 0.45, 0.70),
        IntradayRule(11, 11, "红绿盘修复：弱涨幅但重新靠近VWAP。", 5, "10:00", "14:15", -0.030, 0.020, 0.040, -0.010, 0.006, -0.002, 0.80, 0.45, 0.70),
        IntradayRule(12, 12, "不追高强市场：强宽度下买低位强势修复。", 5, "10:00", "14:15", -0.010, 0.040, 0.060, -0.005, 0.012, -0.002, 0.70, 0.55, 0.70),
    ]


def fit_model(train: pd.DataFrame, model_config: IntradayModel) -> Pipeline:
    sample = train.dropna(subset=[model_config.target]).copy()
    rf = RandomForestRegressor(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        min_samples_leaf=model_config.min_samples_leaf,
        max_features=model_config.max_features,
        max_samples=model_config.max_samples,
        bootstrap=True,
        random_state=model_config.random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", rf)])
    pipe.fit(sample[list(FEATURES)], sample[model_config.target])
    return pipe


def predict(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[list(FEATURES)])
    return out


def apply_rule(predicted: pd.DataFrame, rule: IntradayRule) -> pd.DataFrame:
    frame = predicted.dropna(subset=["prediction", "target_return", "entry_next_open"]).copy()
    wide = is_wide_limit_symbol(frame["symbol"])
    max_ret = np.where(wide, rule.max_ret_prev_close_wide, rule.max_ret_prev_close_main)
    mask = (
        (frame["time"] >= rule.signal_start)
        & (frame["time"] <= rule.signal_end)
        & (frame["ret_prev_close"] >= rule.min_ret_prev_close)
        & (frame["ret_prev_close"] <= max_ret)
        & (frame["ret_from_open"] >= rule.min_ret_from_open)
        & (frame["ret_from_open"] <= rule.max_ret_from_open)
        & (frame["above_vwap"] >= rule.min_above_vwap)
        & (frame["cum_amount_ratio_10"] >= rule.min_cum_amount_ratio)
        & (frame["market_breadth_open"] >= rule.min_market_breadth)
    )
    frame = frame[mask].copy()
    if frame.empty:
        return frame
    threshold = frame["prediction"].quantile(rule.min_prediction_quantile)
    frame = frame[frame["prediction"] >= threshold]
    selected = []
    for date, day in frame.sort_values(["date", "prediction"], ascending=[True, False]).groupby("date"):
        chosen = day.drop_duplicates("symbol").head(rule.top_n).copy()
        chosen["signal_date"] = date
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def backtest(predicted: pd.DataFrame, rule: IntradayRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = apply_rule(predicted, rule)
    dates = pd.Index(sorted(predicted["date"].unique()))
    rows = []
    equity = 1.0
    if trades.empty:
        for date in dates:
            rows.append({"date": date, "net_return": 0.0, "gross_return": 0.0, "n_positions": 0, "equity": equity})
        return pd.DataFrame(rows), trades
    trades["gross_return"] = trades["target_return"]
    trades["net_trade_return"] = trades["gross_return"] - rule.transaction_cost_bps / 10000
    daily = trades.groupby("date").agg(gross_return=("gross_return", "mean"), net_return=("net_trade_return", "mean"), n_positions=("symbol", "nunique")).reset_index()
    daily_by_date = daily.set_index("date")
    for date in dates:
        if date in daily_by_date.index:
            item = daily_by_date.loc[date]
            net_return = float(item["net_return"])
            gross_return = float(item["gross_return"])
            n_positions = int(item["n_positions"])
        else:
            net_return = 0.0
            gross_return = 0.0
            n_positions = 0
        equity *= 1 + net_return
        rows.append({"date": date, "net_return": net_return, "gross_return": gross_return, "n_positions": n_positions, "equity": equity})
    return pd.DataFrame(rows), trades


def metrics(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {"sharpe": 0.0, "annual_return": 0.0, "annual_vol": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "win_rate": 0.0, "avg_positions": 0.0, "trade_days": 0}
    returns = equity["net_return"].fillna(0)
    mean = returns.mean()
    std = returns.std(ddof=0)
    sharpe = 0.0 if std == 0 else float(mean / std * np.sqrt(252))
    total = float(equity["equity"].iloc[-1] - 1)
    annual = float(equity["equity"].iloc[-1] ** (252 / max(1, len(equity))) - 1)
    drawdown = equity["equity"] / equity["equity"].cummax() - 1
    active = returns[returns != 0]
    return {
        "sharpe": sharpe,
        "annual_return": annual,
        "annual_vol": float(std * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "total_return": total,
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "avg_positions": float(equity["n_positions"].mean()),
        "trade_days": int((equity["n_positions"] > 0).sum()),
    }


def split_dates(features: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    dates = pd.Index(sorted(features["date"].dropna().unique()))
    n_train = int(len(dates) * train_ratio)
    n_val = int(len(dates) * val_ratio)
    return dates[:n_train], dates[n_train : n_train + n_val], dates[n_train + n_val :]


def evaluate(features: pd.DataFrame):
    train_dates, val_dates, live_dates = split_dates(features)
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()
    rows = []
    artifacts = {}
    for model_config in models():
        print(f"training model {model_config.model_id}: {model_config.change}", flush=True)
        model = fit_model(train, model_config)
        val_pred = predict(model, val)
        live_pred = predict(model, live)
        for rule in rules():
            if rule.round_id < model_config.round_id - 1:
                continue
            val_bt, val_trades = backtest(val_pred, rule)
            live_bt, live_trades = backtest(live_pred, rule)
            val_m = metrics(val_bt)
            live_m = metrics(live_bt)
            score = val_m["sharpe"] + 0.4 * val_m["win_rate"] - max(0, 3 - val_m["trade_days"]) * 4.0
            row = {
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
            rows.append(row)
            artifacts[(model_config.model_id, rule.rule_id)] = (val_bt, val_trades, live_bt, live_trades)
    candidates = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    return candidates, artifacts, train_dates, val_dates, live_dates


def final_retrain(features: pd.DataFrame, train_dates, val_dates, live_dates, model_config: IntradayModel, rule: IntradayRule):
    train_val = features[features["date"].isin(list(train_dates) + list(val_dates))].copy()
    live = features[features["date"].isin(live_dates)].copy()
    model = fit_model(train_val, model_config)
    live_pred = predict(model, live)
    bt, trades = backtest(live_pred, rule)
    return metrics(bt), bt, trades


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    feature_path = PROCESSED_DIR / f"csi1000_intraday_{args.period}m_features.parquet"
    if feature_path.exists() and not args.refresh_features:
        features = pd.read_parquet(feature_path)
    else:
        symbols = select_liquid_symbols(args.max_symbols)
        symbols.to_csv(UNIVERSE_DIR / "csi1000_intraday_used.csv", index=False)
        bars_path = PROCESSED_DIR / f"csi1000_intraday_{args.period}m_bars.parquet"
        if bars_path.exists() and not args.refresh_data:
            bars = pd.read_parquet(bars_path)
        else:
            bars = load_intraday(symbols, args.start_date, args.end_date, args.period, args.refresh_data)
            bars.to_parquet(bars_path, index=False)
        features = build_features(bars)
        features = features.dropna(subset=["target_return", "prev_close", "cum_amount_ratio_10"]).copy()
        features.to_parquet(feature_path, index=False)

    candidates, artifacts, train_dates, val_dates, live_dates = evaluate(features)
    candidates.to_csv(INTRADAY_REPORT_DIR / "iteration_candidates.csv", index=False)
    top = candidates.head(12)
    top_final_rows = []
    model_by_id = {item.model_id: item for item in models()}
    rule_by_id = {item.rule_id: item for item in rules()}
    for _, row in top.iterrows():
        model_config = model_by_id[int(row["model_id"])]
        rule = rule_by_id[int(row["rule_id"])]
        final_m, final_bt, final_trades = final_retrain(features, train_dates, val_dates, live_dates, model_config, rule)
        tag = f"candidate_{model_config.model_id}_{rule.rule_id}"
        final_bt.to_csv(INTRADAY_REPORT_DIR / f"{tag}_equity.csv", index=False)
        final_trades.to_csv(INTRADAY_REPORT_DIR / f"{tag}_trades.csv", index=False)
        top_final_rows.append({**row.to_dict(), **{f"final_{k}": v for k, v in final_m.items()}})
    top_final = pd.DataFrame(top_final_rows).sort_values("final_sharpe", ascending=False)
    top_final.to_csv(INTRADAY_REPORT_DIR / "top_final_checks.csv", index=False)
    best = top_final.iloc[0].to_dict()
    best_model = model_by_id[int(best["model_id"])]
    best_rule = rule_by_id[int(best["rule_id"])]
    best_m, best_bt, best_trades = final_retrain(features, train_dates, val_dates, live_dates, best_model, best_rule)
    best_bt.to_csv(INTRADAY_REPORT_DIR / "best_equity.csv", index=False)
    best_trades.to_csv(INTRADAY_REPORT_DIR / "best_trades.csv", index=False)
    summary = {
        "objective": "A-share intraday tradable momentum: 15m signal, next-bar entry, T+1 overnight exit.",
        "dataset": {
            "symbols": int(features["symbol"].nunique()),
            "rows": int(len(features)),
            "start": str(features["datetime"].min()),
            "end": str(features["datetime"].max()),
            "dates": int(features["date"].nunique()),
        },
        "split": {
            "train": f"{train_dates.min()} to {train_dates.max()} ({len(train_dates)} dates)",
            "validation": f"{val_dates.min()} to {val_dates.max()} ({len(val_dates)} dates)",
            "paper_live": f"{live_dates.min()} to {live_dates.max()} ({len(live_dates)} dates)",
        },
        "best_by_final_check": clean_for_json(best),
        "best_model": asdict(best_model),
        "best_rule": asdict(best_rule),
        "best_final_metrics": best_m,
        "top_final_checks": clean_for_json(top_final.head(12).to_dict(orient="records")),
    }
    (INTRADAY_REPORT_DIR / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(top_final[["model_id", "rule_id", "round_id", "validation_sharpe", "validation_trade_days", "paper_live_sharpe", "paper_live_trade_days", "final_sharpe", "final_trade_days", "final_total_return", "final_max_drawdown"]].to_string(index=False))
    print("\nBest model:", best_model)
    print("Best rule:", best_rule)
    print("Best final metrics:")
    print(json.dumps(clean_for_json(best_m), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20250101")
    parser.add_argument("--end-date", default="20260610")
    parser.add_argument("--period", default="15")
    parser.add_argument("--max-symbols", type=int, default=80)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
