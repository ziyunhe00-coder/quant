from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rf_strategy_search import (
    ModelSpec,
    PortfolioRule,
    backtest,
    enrich_features,
    fit_model,
    period_metrics,
    predict,
)
from random_forest_a_share_strategy import (
    PROCESSED_DIR,
    REPORT_DIR,
    clean_for_json,
    ensure_dirs,
    normalize_stock_hist,
    split_dates,
)


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_DIR = ROOT / "data" / "universe"
CSI500_RAW_DIR = ROOT / "data" / "raw_csi500"
CSI500_BAOSTOCK_RAW_DIR = ROOT / "data" / "raw_csi500_baostock"
CSI500_YF_RAW_DIR = ROOT / "data" / "raw_csi500_yfinance"
SP500_RAW_DIR = ROOT / "data" / "raw_sp500"
LARGE_REPORT_DIR = REPORT_DIR / "large_universe"


FIXED_MODEL = ModelSpec(
    model_id=1,
    reason="Fixed from 20-stock adaptive search: shallow RF, next-day absolute return.",
    target="return_1",
    features=(
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
    ),
    n_estimators=350,
    max_depth=4,
    min_samples_leaf=25,
    max_features="sqrt",
)

FIXED_RULE = PortfolioRule(
    rule_id=43,
    reason="Fixed from 20-stock adaptive search: breadth > 50%, Top1, 10-day rebalance.",
    direction="highest",
    top_n=1,
    rebalance_every=10,
    min_pred_quantile=None,
    market_filter="breadth_above_50",
    stock_filter="none",
    weighting="equal",
    transaction_cost_bps=12.0,
)


def ensure_large_dirs() -> None:
    ensure_dirs()
    for path in (UNIVERSE_DIR, CSI500_RAW_DIR, CSI500_BAOSTOCK_RAW_DIR, CSI500_YF_RAW_DIR, SP500_RAW_DIR, LARGE_REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def a_share_csi500_universe(refresh: bool = False) -> pd.DataFrame:
    cache_path = UNIVERSE_DIR / "csi500_constituents.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, dtype={"code": str})

    import akshare as ak

    df = ak.index_stock_cons_sina(symbol="000905")
    out = df[["code", "name"]].copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out = out.drop_duplicates("code").sort_values("code")
    out.to_csv(cache_path, index=False)
    return out


def fetch_a_share_history(symbol: str, name: str, start_date: str, end_date: str, refresh: bool) -> pd.DataFrame | None:
    cache_path = CSI500_RAW_DIR / f"{symbol}_{start_date}_{end_date}_qfq.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["date"], dtype={"symbol": str})

    import akshare as ak

    def _timeout_handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"timeout fetching {symbol}")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        for attempt in range(3):
            try:
                signal.alarm(25)
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )
                signal.alarm(0)
                out = normalize_stock_hist(df, symbol, name)
                out.to_csv(cache_path, index=False)
                time.sleep(0.15)
                return out
            except Exception as exc:  # noqa: BLE001
                signal.alarm(0)
                if attempt == 2:
                    print(f"[a-share] failed {symbol} {name}: {exc}")
                    return None
                time.sleep(1.0 + attempt)
            finally:
                signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
    return None


def baostock_symbol(symbol: str) -> str:
    return f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"


def fetch_a_share_history_baostock(symbol: str, name: str, start_date: str, end_date: str, refresh: bool) -> pd.DataFrame | None:
    cache_path = CSI500_BAOSTOCK_RAW_DIR / f"{symbol}_{start_date}_{end_date}_qfq.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["date"], dtype={"symbol": str})

    import baostock as bs

    fields = "date,code,open,high,low,close,volume,amount,turn"
    rs = bs.query_history_k_data_plus(
        baostock_symbol(symbol),
        fields,
        start_date=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
        end_date=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
        frequency="d",
        adjustflag="2",
    )
    if rs.error_code != "0":
        print(f"[a-share/baostock] failed {symbol} {name}: {rs.error_msg}")
        return None

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None

    out = pd.DataFrame(rows, columns=rs.fields)
    out = out.rename(columns={"turn": "turnover_rate"})
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out["symbol"] = symbol
    out["name"] = name
    out = out[["date", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "symbol", "name"]]
    out.to_csv(cache_path, index=False)
    return out.sort_values("date")


def load_a_share_csi500(start_date: str, end_date: str, refresh: bool, max_symbols: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = a_share_csi500_universe(refresh=refresh)
    if max_symbols:
        universe = universe.head(max_symbols)

    yf_universe = universe.rename(columns={"code": "symbol"}).copy()
    yf_universe["yf_symbol"] = yf_universe["symbol"].map(
        lambda code: f"{code}.SS" if str(code).startswith("6") else f"{code}.SZ"
    )
    frames = []
    batch_size = 50
    for start in range(0, len(yf_universe), batch_size):
        batch = yf_universe.iloc[start : start + batch_size]
        frames.extend(fetch_yfinance_batch(batch, CSI500_YF_RAW_DIR, start_date, end_date, refresh, label="a-share/yfinance"))
        print(f"[a-share/yfinance] loaded {len(frames)}/{min(start + batch_size, len(yf_universe))} symbols")
        time.sleep(0.5)

    if len(frames) < 50:
        raise RuntimeError(f"Only loaded {len(frames)} CSI500 symbols; too few for large-universe validation.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]), universe


def sp500_universe(refresh: bool = False) -> pd.DataFrame:
    cache_path = UNIVERSE_DIR / "sp500_constituents.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    import requests

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 quant-research/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0][["Symbol", "Security"]].copy()
    df.columns = ["symbol", "name"]
    df["yf_symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    df = df.drop_duplicates("symbol").sort_values("symbol")
    df.to_csv(cache_path, index=False)
    return df


def normalize_yfinance_frame(df: pd.DataFrame, symbol: str, name: str) -> pd.DataFrame | None:
    if df.empty:
        return None
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
    }
    out = df.rename(columns=rename).reset_index()
    date_col = "Date" if "Date" in out.columns else "date"
    out = out.rename(columns={date_col: "date"})
    needed = ["date", "open", "high", "low", "close", "volume"]
    if any(col not in out.columns for col in needed):
        return None
    out = out[needed].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    out["amount"] = out["close"] * out["volume"]
    out["symbol"] = symbol
    out["name"] = name
    return out.sort_values("date")


def fetch_yfinance_batch(
    batch: pd.DataFrame,
    raw_dir: Path,
    start_date: str,
    end_date: str,
    refresh: bool,
    label: str,
) -> list[pd.DataFrame]:
    import yfinance as yf

    frames = []
    missing = []
    to_download = []
    rows_by_yf = {}
    for row in batch.itertuples(index=False):
        cache_path = raw_dir / f"{row.symbol}_{start_date}_{end_date}.csv"
        if cache_path.exists() and not refresh:
            frame = pd.read_csv(cache_path, parse_dates=["date"])
            frames.append(frame)
        else:
            to_download.append(row.yf_symbol)
            rows_by_yf[row.yf_symbol] = row

    if not to_download:
        return frames

    data = yf.download(
        tickers=to_download,
        start=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
        end=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    for yf_symbol in to_download:
        row = rows_by_yf[yf_symbol]
        try:
            ticker_df = data[yf_symbol] if isinstance(data.columns, pd.MultiIndex) else data
            out = normalize_yfinance_frame(ticker_df, row.symbol, row.name)
            if out is not None and len(out) >= 260:
                out.to_csv(raw_dir / f"{row.symbol}_{start_date}_{end_date}.csv", index=False)
                frames.append(out)
            else:
                missing.append(row.symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] failed {row.symbol}: {exc}")
            missing.append(row.symbol)
    if missing:
        print(f"[{label}] missing in batch: {','.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
    return frames


def fetch_sp500_batch(batch: pd.DataFrame, start_date: str, end_date: str, refresh: bool) -> list[pd.DataFrame]:
    return fetch_yfinance_batch(batch, SP500_RAW_DIR, start_date, end_date, refresh, label="sp500")


def load_sp500(start_date: str, end_date: str, refresh: bool, max_symbols: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = sp500_universe(refresh=refresh)
    if max_symbols:
        universe = universe.head(max_symbols)

    frames = []
    batch_size = 50
    for start in range(0, len(universe), batch_size):
        batch = universe.iloc[start : start + batch_size]
        frames.extend(fetch_sp500_batch(batch, start_date, end_date, refresh))
        print(f"[sp500] loaded {len(frames)}/{min(start + batch_size, len(universe))} symbols")
        time.sleep(0.5)

    if len(frames) < 50:
        raise RuntimeError(f"Only loaded {len(frames)} S&P 500 symbols; too few for large-universe validation.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]), universe


def run_fixed_strategy(prices: pd.DataFrame, dataset_name: str) -> dict:
    features = enrich_features(prices).dropna(subset=["fwd_return_1"]).copy()
    features.to_parquet(PROCESSED_DIR / f"{dataset_name}_features.parquet", index=False)

    split_basis = features.dropna(subset=["ret_20", "vol_20", "ma_gap_20"])
    train_dates, val_dates, live_dates = split_dates(split_basis)
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()

    train_model = fit_model(train, FIXED_MODEL)
    val_pred = predict(train_model, val, FIXED_MODEL)
    live_pred_train_only = predict(train_model, live, FIXED_MODEL)
    val_bt, val_holdings = backtest(val_pred, FIXED_RULE)
    live_bt_train_only, live_holdings_train_only = backtest(live_pred_train_only, FIXED_RULE)

    train_val = features[features["date"].isin(list(train_dates) + list(val_dates))].copy()
    final_model = fit_model(train_val, FIXED_MODEL)
    final_live_pred = predict(final_model, live, FIXED_MODEL)
    final_bt, final_holdings = backtest(final_live_pred, FIXED_RULE)

    prefix = LARGE_REPORT_DIR / dataset_name
    val_bt.to_csv(prefix.with_name(f"{dataset_name}_validation_equity.csv"), index=False)
    val_holdings.to_csv(prefix.with_name(f"{dataset_name}_validation_holdings.csv"), index=False)
    live_bt_train_only.to_csv(prefix.with_name(f"{dataset_name}_paper_live_train_only_equity.csv"), index=False)
    live_holdings_train_only.to_csv(prefix.with_name(f"{dataset_name}_paper_live_train_only_holdings.csv"), index=False)
    final_bt.to_csv(prefix.with_name(f"{dataset_name}_final_paper_live_equity.csv"), index=False)
    final_holdings.to_csv(prefix.with_name(f"{dataset_name}_final_paper_live_holdings.csv"), index=False)

    return {
        "dataset": dataset_name,
        "symbols_loaded": int(prices["symbol"].nunique()),
        "rows": int(len(prices)),
        "feature_rows": int(len(features)),
        "date_range": {
            "start": str(prices["date"].min().date()),
            "end": str(prices["date"].max().date()),
        },
        "split": {
            "train": f"{pd.DatetimeIndex(train_dates).min().date()} to {pd.DatetimeIndex(train_dates).max().date()} ({len(train_dates)} dates)",
            "validation": f"{pd.DatetimeIndex(val_dates).min().date()} to {pd.DatetimeIndex(val_dates).max().date()} ({len(val_dates)} dates)",
            "paper_live": f"{pd.DatetimeIndex(live_dates).min().date()} to {pd.DatetimeIndex(live_dates).max().date()} ({len(live_dates)} dates)",
        },
        "model": asdict(FIXED_MODEL),
        "rule": asdict(FIXED_RULE),
        "validation": period_metrics(val_bt),
        "paper_live_train_only": period_metrics(live_bt_train_only),
        "final_paper_live_retrained_on_90pct": period_metrics(final_bt),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["ashare-csi500", "sp500", "both"], default="both")
    parser.add_argument("--start-date", default="20180101")
    parser.add_argument("--end-date", default="20260610")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_large_dirs()
    summaries = []

    if args.market in {"ashare-csi500", "both"}:
        prices, universe = load_a_share_csi500(args.start_date, args.end_date, args.refresh, args.max_symbols)
        universe.to_csv(UNIVERSE_DIR / "csi500_used.csv", index=False)
        summaries.append(run_fixed_strategy(prices, "ashare_csi500"))

    if args.market in {"sp500", "both"}:
        prices, universe = load_sp500(args.start_date, args.end_date, args.refresh, args.max_symbols)
        universe.to_csv(UNIVERSE_DIR / "sp500_used.csv", index=False)
        summaries.append(run_fixed_strategy(prices, "sp500"))

    summary = {
        "strategy_note": "Fixed strategy selected on original 20-stock A-share experiment; no retuning on large universes.",
        "results": summaries,
    }
    (LARGE_REPORT_DIR / "large_universe_fixed_strategy_summary.json").write_text(
        json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for item in summaries:
        print("\n==", item["dataset"], "==")
        print("symbols:", item["symbols_loaded"], "rows:", item["rows"])
        print("validation sharpe:", item["validation"]["sharpe"])
        print("train-only paper-live sharpe:", item["paper_live_train_only"]["sharpe"])
        print("final paper-live sharpe:", item["final_paper_live_retrained_on_90pct"]["sharpe"])
        print("final max drawdown:", item["final_paper_live_retrained_on_90pct"]["max_drawdown"])


if __name__ == "__main__":
    main()
