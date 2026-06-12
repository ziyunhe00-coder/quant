from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


UNIVERSE = {
    "000001": "平安银行",
    "000333": "美的集团",
    "000651": "格力电器",
    "000725": "京东方A",
    "000858": "五粮液",
    "002415": "海康威视",
    "002594": "比亚迪",
    "300750": "宁德时代",
    "300760": "迈瑞医疗",
    "600030": "中信证券",
    "600036": "招商银行",
    "600276": "恒瑞医药",
    "600309": "万华化学",
    "600519": "贵州茅台",
    "600887": "伊利股份",
    "600900": "长江电力",
    "601012": "隆基绿能",
    "601166": "兴业银行",
    "601318": "中国平安",
    "601888": "中国中免",
}


BASE_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_10",
    "vol_20",
    "ma_gap_20",
    "amount_z20",
]


@dataclass(frozen=True)
class IterationConfig:
    iteration: int
    change: str
    features: tuple[str, ...]
    target_mode: str = "return"
    n_estimators: int = 350
    max_depth: int | None = 5
    min_samples_leaf: int = 25
    max_features: str | float = "sqrt"
    top_n: int = 5
    rebalance_every: int = 1
    min_pred_quantile: float | None = None
    transaction_cost_bps: float = 12.0
    random_state: int = 42


ITERATIONS = [
    IterationConfig(
        iteration=1,
        change="基线：动量、波动、均线偏离、成交额特征；日频调仓，Top5。",
        features=tuple(BASE_FEATURES),
    ),
    IterationConfig(
        iteration=2,
        change="增加短周期反转/RSI/价格振幅特征，并略放深树。",
        features=tuple(BASE_FEATURES + ["ret_3", "rsi_14", "range_5"]),
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=18,
    ),
    IterationConfig(
        iteration=3,
        change="将训练目标改成每日横截面收益排名，降低市场整体涨跌对模型的干扰。",
        features=tuple(BASE_FEATURES + ["ret_3", "rsi_14", "range_5"]),
        target_mode="rank",
        n_estimators=550,
        max_depth=6,
        min_samples_leaf=18,
    ),
    IterationConfig(
        iteration=4,
        change="在第3轮基础上降低换手：每5个交易日调仓一次，并集中到Top3。",
        features=tuple(BASE_FEATURES + ["ret_3", "rsi_14", "range_5"]),
        target_mode="rank",
        n_estimators=550,
        max_depth=6,
        min_samples_leaf=18,
        top_n=3,
        rebalance_every=5,
    ),
    IterationConfig(
        iteration=5,
        change="加入预测阈值过滤：只在预测值超过当日60%分位时持仓，控制弱信号交易。",
        features=tuple(BASE_FEATURES + ["ret_3", "rsi_14", "range_5", "ma_gap_60", "vol_ratio_5_20"]),
        target_mode="rank",
        n_estimators=650,
        max_depth=7,
        min_samples_leaf=15,
        top_n=3,
        rebalance_every=5,
        min_pred_quantile=0.60,
    ),
]


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_stock_hist(df: pd.DataFrame, symbol: str, name: str) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns=rename)
    needed = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{symbol} missing columns: {missing}")
    out = df[needed + ([ "turnover_rate" ] if "turnover_rate" in df.columns else [])].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["symbol"] = symbol
    out["name"] = name
    return out.sort_values("date")


def fetch_one_stock(symbol: str, name: str, start_date: str, end_date: str, refresh: bool) -> pd.DataFrame:
    cache_path = RAW_DIR / f"{symbol}_{start_date}_{end_date}_qfq.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["date"])

    import akshare as ak

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    out = normalize_stock_hist(df, symbol, name)
    out.to_csv(cache_path, index=False)
    time.sleep(0.25)
    return out


def load_market_data(start_date: str, end_date: str, refresh: bool) -> pd.DataFrame:
    frames = []
    failures = []
    failure_path = REPORT_DIR / "data_failures.csv"
    for symbol, name in UNIVERSE.items():
        try:
            frame = fetch_one_stock(symbol, name, start_date, end_date, refresh)
            if len(frame) >= 260:
                frames.append(frame)
            else:
                failures.append((symbol, name, "history shorter than 260 rows"))
        except Exception as exc:  # noqa: BLE001
            failures.append((symbol, name, str(exc)))

    if failures:
        pd.DataFrame(failures, columns=["symbol", "name", "reason"]).to_csv(failure_path, index=False)
    elif failure_path.exists():
        failure_path.unlink()

    if len(frames) < 8:
        raise RuntimeError(
            f"Only {len(frames)} symbols loaded. Need at least 8 for a cross-sectional strategy. "
            f"See {REPORT_DIR / 'data_failures.csv'} for details."
        )
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in prices.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        ret = group["close"].pct_change()
        group["ret_1"] = ret
        group["ret_3"] = group["close"].pct_change(3)
        group["ret_5"] = group["close"].pct_change(5)
        group["ret_20"] = group["close"].pct_change(20)
        group["vol_10"] = ret.rolling(10).std() * math.sqrt(252)
        group["vol_20"] = ret.rolling(20).std() * math.sqrt(252)
        group["ma_gap_20"] = group["close"] / group["close"].rolling(20).mean() - 1
        group["ma_gap_60"] = group["close"] / group["close"].rolling(60).mean() - 1
        group["amount_z20"] = zscore(np.log1p(group["amount"]), 20)
        group["rsi_14"] = rsi(group["close"], 14)
        group["range_5"] = ((group["high"] - group["low"]) / group["close"]).rolling(5).mean()
        group["vol_ratio_5_20"] = (
            ret.rolling(5).std() / ret.rolling(20).std().replace(0, np.nan)
        )
        group["fwd_return_1"] = group["close"].pct_change().shift(-1)
        pieces.append(group)

    features = pd.concat(pieces, ignore_index=True)
    features["target_rank"] = features.groupby("date")["fwd_return_1"].rank(pct=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def split_dates(df: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = pd.DatetimeIndex(sorted(df["date"].dropna().unique()))
    train_end = int(len(dates) * 0.80)
    val_end = int(len(dates) * 0.90)
    if train_end <= 0 or val_end <= train_end or len(dates) - val_end <= 0:
        raise RuntimeError("Not enough dates for 80/10/10 split.")
    return dates[:train_end], dates[train_end:val_end], dates[val_end:]


def target_column(mode: str) -> str:
    if mode == "return":
        return "fwd_return_1"
    if mode == "rank":
        return "target_rank"
    raise ValueError(f"Unknown target_mode: {mode}")


def fit_model(train: pd.DataFrame, config: IterationConfig) -> Pipeline:
    col = target_column(config.target_mode)
    sample = train.dropna(subset=[col]).copy()
    x = sample[list(config.features)]
    y = sample[col]
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
        n_jobs=-1,
        bootstrap=True,
    )
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipe.fit(x, y)
    return pipe


def predict_frame(model: Pipeline, df: pd.DataFrame, config: IterationConfig) -> pd.DataFrame:
    out = df.copy()
    out["prediction"] = model.predict(out[list(config.features)])
    return out


def selected_weights(day: pd.DataFrame, config: IterationConfig) -> dict[str, float]:
    day = day.dropna(subset=["prediction", "fwd_return_1"]).sort_values("prediction", ascending=False)
    if day.empty:
        return {}

    if config.min_pred_quantile is not None:
        threshold = day["prediction"].quantile(config.min_pred_quantile)
        day = day[day["prediction"] >= threshold]

    day = day.head(config.top_n)
    if day.empty:
        return {}
    weight = 1.0 / len(day)
    return {symbol: weight for symbol in day["symbol"].tolist()}


def portfolio_return(day: pd.DataFrame, weights: dict[str, float]) -> float:
    if not weights:
        return 0.0
    returns = day.set_index("symbol")["fwd_return_1"]
    value = 0.0
    for symbol, weight in weights.items():
        item_return = returns.get(symbol, np.nan)
        if pd.notna(item_return):
            value += weight * float(item_return)
    return value


def turnover(prev: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(prev) | set(current)
    return sum(abs(current.get(symbol, 0.0) - prev.get(symbol, 0.0)) for symbol in symbols)


def backtest(predicted: pd.DataFrame, config: IterationConfig) -> pd.DataFrame:
    rows = []
    previous_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}

    for i, (date, day) in enumerate(predicted.groupby("date", sort=True)):
        rebalanced = i % config.rebalance_every == 0
        day_turnover = 0.0
        if rebalanced:
            current_weights = selected_weights(day, config)
            day_turnover = turnover(previous_weights, current_weights)
            previous_weights = current_weights

        gross_return = portfolio_return(day, current_weights)
        cost = day_turnover * config.transaction_cost_bps / 10000
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": gross_return - cost,
                "n_positions": len(current_weights),
                "turnover": day_turnover,
            }
        )

    return pd.DataFrame(rows)


def metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.fillna(0.0)
    if returns.empty:
        return {
            "sharpe": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }
    mean = returns.mean()
    std = returns.std(ddof=1)
    sharpe = 0.0 if std == 0 or pd.isna(std) else float(mean / std * math.sqrt(252))
    equity = (1 + returns).cumprod()
    annual_return = float(equity.iloc[-1] ** (252 / len(returns)) - 1)
    annual_vol = float(std * math.sqrt(252)) if pd.notna(std) else 0.0
    drawdown = equity / equity.cummax() - 1
    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity.iloc[-1] - 1),
    }


def run_iteration(
    features: pd.DataFrame,
    train_dates: Iterable[pd.Timestamp],
    val_dates: Iterable[pd.Timestamp],
    live_dates: Iterable[pd.Timestamp],
    config: IterationConfig,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()
    model = fit_model(train, config)
    val_bt = backtest(predict_frame(model, val, config), config)
    live_bt = backtest(predict_frame(model, live, config), config)
    val_metrics = metrics(val_bt["net_return"])
    live_metrics = metrics(live_bt["net_return"])

    row = {
        "iteration": config.iteration,
        "change": config.change,
        "target_mode": config.target_mode,
        "features": ",".join(config.features),
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "top_n": config.top_n,
        "rebalance_every": config.rebalance_every,
        "min_pred_quantile": config.min_pred_quantile,
        "transaction_cost_bps": config.transaction_cost_bps,
        "validation_sharpe": val_metrics["sharpe"],
        "validation_annual_return": val_metrics["annual_return"],
        "validation_max_drawdown": val_metrics["max_drawdown"],
        "paper_live_sharpe_train_only": live_metrics["sharpe"],
        "paper_live_annual_return_train_only": live_metrics["annual_return"],
        "paper_live_max_drawdown_train_only": live_metrics["max_drawdown"],
    }
    return row, val_bt, live_bt


def final_holdout_backtest(
    features: pd.DataFrame,
    train_dates: Iterable[pd.Timestamp],
    val_dates: Iterable[pd.Timestamp],
    live_dates: Iterable[pd.Timestamp],
    config: IterationConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_val_dates = list(train_dates) + list(val_dates)
    train_val = features[features["date"].isin(train_val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()
    model = fit_model(train_val, config)
    live_bt = backtest(predict_frame(model, live, config), config)
    return live_bt, metrics(live_bt["net_return"])


def date_range_text(dates: Iterable[pd.Timestamp]) -> str:
    dates = pd.DatetimeIndex(dates)
    return f"{dates.min().date()} to {dates.max().date()} ({len(dates)} trading signal dates)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20180101")
    parser.add_argument("--end-date", default=datetime.today().strftime("%Y%m%d"))
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def clean_for_json(value):
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value) or not np.isfinite(value):
            return None
        return float(value)
    return value


def main() -> None:
    args = parse_args()
    ensure_dirs()

    raw = load_market_data(args.start_date, args.end_date, args.refresh)
    features = build_features(raw).dropna(subset=["fwd_return_1"]).copy()
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    train_dates, val_dates, live_dates = split_dates(features.dropna(subset=BASE_FEATURES))
    split_info = {
        "train": date_range_text(train_dates),
        "validation": date_range_text(val_dates),
        "paper_live": date_range_text(live_dates),
    }

    rows = []
    for config in ITERATIONS:
        row, val_bt, live_bt = run_iteration(features, train_dates, val_dates, live_dates, config)
        rows.append(row)
        val_bt.to_csv(REPORT_DIR / f"iteration_{config.iteration}_validation_equity.csv", index=False)
        live_bt.to_csv(REPORT_DIR / f"iteration_{config.iteration}_paper_live_train_only_equity.csv", index=False)

    log = pd.DataFrame(rows).sort_values("iteration")
    log.to_csv(REPORT_DIR / "iteration_log.csv", index=False)

    best_row = log.sort_values("validation_sharpe", ascending=False).iloc[0].to_dict()
    best_config = next(config for config in ITERATIONS if config.iteration == int(best_row["iteration"]))
    final_bt, final_metrics = final_holdout_backtest(features, train_dates, val_dates, live_dates, best_config)
    final_bt.to_csv(REPORT_DIR / "final_paper_live_equity.csv", index=False)

    summary = {
        "data_source": "AkShare stock_zh_a_hist, qfq adjusted daily bars",
        "universe": UNIVERSE,
        "split": split_info,
        "selection_standard": "highest validation_sharpe",
        "best_iteration": int(best_config.iteration),
        "best_change": best_config.change,
        "best_config": asdict(best_config),
        "best_validation": best_row,
        "final_paper_live_retrained_on_90pct": final_metrics,
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Split:")
    for key, value in split_info.items():
        print(f"  {key}: {value}")
    print("\nIteration log:")
    print(
        log[
            [
                "iteration",
                "validation_sharpe",
                "paper_live_sharpe_train_only",
                "change",
            ]
        ].to_string(index=False)
    )
    print("\nBest iteration by validation Sharpe:", best_config.iteration)
    print("Final paper-live metrics after retraining on first 90%:")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
