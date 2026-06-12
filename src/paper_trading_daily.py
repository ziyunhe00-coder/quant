from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from csi_fundamental_improvement_analysis import (
    attach_industry,
    columns_for_spec,
    diversified_rules,
    fit_model,
    model_specs,
    predict_frame,
    select_diversified_weights,
)
from csi_fundamental_regime_exposure import target_exposure
from csi_fundamental_recency_weighted_validation import fit_weighted_model, recency_schemes
from csi_fundamental_true_sharpe_validation import exposure_candidates
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_strategy import metrics
from random_forest_a_share_strategy import clean_for_json


def load_config(path: Path) -> dict:
    defaults = {
        "max_symbols": 200,
        "capital": 100000,
        "lot_size": 100,
        "industry_map": "data/processed/csi1000_industry_map_200.csv",
        "industry_col": "industry_gate",
        "output_dir": "paper_trading",
        "positions_path": "paper_trading/positions.csv",
        "train_min": 760,
        "val_len": 194,
        "embargo": 40,
        "score_mode": "sharpe",
        "vol_budgets": "0.18,0.20,0.22,0.24,0.26,0.28,0.30",
        "recency_schemes": "full_equal,exp_hl_126,exp_hl_252,exp_hl_504,rolling_504",
        "model_ids": "1,3,4",
        "trade_rule_ids": "401,402,403",
        "exposure_names": "vol_0.18,vol_0.26,vol_0.30,trend_vol_0.18,trend_vol_0.26,trend_breadth_keep",
        "min_price": 2,
        "max_price_for_new_buy": 300,
        "min_order_value": 1000,
        "send_empty_orders": False,
    }
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            defaults.update(json.load(f))
    return defaults


def regime_score(m: dict[str, float], score_mode: str) -> float:
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


def scale_to_exposure(weights: dict[str, float], gross_limit: float) -> dict[str, float]:
    gross = sum(abs(value) for value in weights.values())
    if gross <= 0 or gross <= gross_limit:
        return weights
    return {symbol: weight * gross_limit / gross for symbol, weight in weights.items()}


def validation_backtest(predicted: pd.DataFrame, trade_rule, exposure_rule) -> pd.DataFrame:
    from csi_fundamental_regime_exposure import dynamic_backtest

    bt, _ = dynamic_backtest(predicted, trade_rule, exposure_rule)
    return bt


def choose_candidate(train: pd.DataFrame, val: pd.DataFrame, features: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    vol_budgets = [float(item) for item in str(config["vol_budgets"]).split(",")]
    exp_rules = exposure_candidates(vol_budgets)
    if config.get("exposure_names"):
        exposure_names = {item.strip() for item in str(config["exposure_names"]).split(",") if item.strip()}
        exp_rules = [rule for rule in exp_rules if rule.name in exposure_names]
    schemes = recency_schemes()
    if config.get("recency_schemes"):
        scheme_names = {item.strip() for item in str(config["recency_schemes"]).split(",") if item.strip()}
        schemes = [scheme for scheme in schemes if scheme.name in scheme_names]
    model_ids = {int(item) for item in str(config.get("model_ids", "")).split(",") if item.strip()}
    trade_rule_ids = {int(item) for item in str(config.get("trade_rule_ids", "")).split(",") if item.strip()}
    specs = model_specs(features)
    if model_ids:
        specs = [spec for spec in specs if spec.model_id in model_ids]
    trade_rules = diversified_rules(constrained_only=True)
    if trade_rule_ids:
        trade_rules = [rule for rule in trade_rules if rule.rule_id in trade_rule_ids]
    rows = []
    best = None
    for scheme in schemes:
        for spec in specs:
            cols = columns_for_spec(features, spec)
            model = fit_weighted_model(train, spec, cols, scheme)
            if model is None:
                continue
            val_pred = predict_frame(model, val, cols)
            for trade_rule in trade_rules:
                for exp_rule in exp_rules:
                    bt = validation_backtest(val_pred, trade_rule, exp_rule)
                    m = metrics(bt)
                    score = regime_score(m, str(config["score_mode"]))
                    row = {
                        "scheme_id": scheme.scheme_id,
                        "scheme_name": scheme.name,
                        "model_id": spec.model_id,
                        "feature_set": spec.feature_set,
                        "trade_rule_id": trade_rule.rule_id,
                        "exposure_name": exp_rule.name,
                        "target_vol": exp_rule.target_vol,
                        "score": score,
                        **{f"validation_{key}": value for key, value in m.items()},
                    }
                    rows.append(row)
                    if best is None or score > best["score"]:
                        best = {
                            **row,
                            "scheme": scheme,
                            "model": spec,
                            "cols": cols,
                            "trade_rule": trade_rule,
                            "exposure_rule": exp_rule,
                        }
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def split_for_live(features: pd.DataFrame, as_of: pd.Timestamp, config: dict):
    eligible = features[(features["date"] <= as_of) & features["ret_60"].notna() & features["fwd_return_40"].notna()].copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    val_len = int(config["val_len"])
    train_min = int(config["train_min"])
    embargo = int(config["embargo"])
    if len(dates) < train_min + val_len + embargo:
        raise ValueError(f"Not enough labeled dates: {len(dates)}")
    val_end = len(dates)
    val_start = val_end - val_len
    train_end = val_start - embargo
    if train_end < train_min:
        raise ValueError(f"Not enough training dates after embargo: train_end={train_end}")
    return dates[:train_end], dates[val_start:val_end]


def load_positions(path: Path) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path, dtype={"symbol": str})
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        return frame
    return pd.DataFrame(columns=["symbol", "shares"])


def make_orders(targets: pd.DataFrame, positions: pd.DataFrame, config: dict) -> pd.DataFrame:
    capital = float(config["capital"])
    lot = int(config["lot_size"])
    current = positions.set_index("symbol")["shares"].to_dict() if not positions.empty else {}
    rows = []
    for _, row in targets.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        close = float(row["close"])
        target_value = float(row["target_weight"]) * capital
        raw_shares = math.floor(target_value / close / lot) * lot
        target_shares = max(0, int(raw_shares))
        if close > float(config["max_price_for_new_buy"]) and current.get(symbol, 0) <= 0:
            target_shares = 0
        current_shares = int(current.get(symbol, 0))
        delta = target_shares - current_shares
        order_value = abs(delta) * close
        if delta != 0 and order_value >= float(config["min_order_value"]):
            rows.append(
                {
                    "symbol": symbol,
                    "name": row.get("name", ""),
                    "side": "BUY" if delta > 0 else "SELL",
                    "shares": abs(delta),
                    "estimated_price": close,
                    "estimated_value": order_value,
                    "target_weight": row["target_weight"],
                    "current_shares": current_shares,
                    "target_shares": target_shares,
                    "prediction": row["prediction"],
                    "industry": row.get(config["industry_col"], ""),
                }
            )
    target_symbols = set(targets["symbol"].astype(str).str.zfill(6))
    for symbol, current_shares in current.items():
        if symbol not in target_symbols and int(current_shares) > 0:
            rows.append(
                {
                    "symbol": symbol,
                    "name": "",
                    "side": "SELL",
                    "shares": int(current_shares),
                    "estimated_price": np.nan,
                    "estimated_value": np.nan,
                    "target_weight": 0.0,
                    "current_shares": int(current_shares),
                    "target_shares": 0,
                    "prediction": np.nan,
                    "industry": "",
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    if args.capital is not None:
        config["capital"] = args.capital
    features = load_cached_features(int(config["max_symbols"]))
    features = attach_industry(features, Path(config["industry_map"]), str(config["industry_col"]))
    features["date"] = pd.to_datetime(features["date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else features["date"].max()
    live_day = features[features["date"] == as_of].copy()
    if live_day.empty:
        raise ValueError(f"No rows for as_of={as_of.date()}")
    live_day = live_day[(live_day["close"] >= float(config["min_price"])) & (live_day["amount"] > 0)].copy()
    train_dates, val_dates = split_for_live(features, as_of, config)
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    best, candidates = choose_candidate(train, val, features, config)

    train_full = features[features["date"] <= val_dates[-1]].copy()
    model = fit_weighted_model(train_full, best["model"], best["cols"], best["scheme"])
    if model is None:
        raise RuntimeError(f"Could not fit selected recency scheme: {best['scheme'].name}")
    live_pred = predict_frame(model, live_day, best["cols"])
    # The selector only requires this column for schema compatibility; live signal generation must not inspect it.
    live_pred["fwd_return_1"] = 0.0

    raw_weights = select_diversified_weights(live_pred, best["trade_rule"])
    desired_exposure = target_exposure(live_pred, best["exposure_rule"], 0.0, best["trade_rule"].max_gross_exposure)
    target_weights = scale_to_exposure(raw_weights, desired_exposure)
    target_frame = (
        live_pred[live_pred["symbol"].isin(target_weights)]
        .copy()
        .assign(target_weight=lambda frame: frame["symbol"].map(target_weights))
        .sort_values("target_weight", ascending=False)
    )
    positions = load_positions(Path(config["positions_path"]))
    orders = make_orders(target_frame, positions, config)

    run_dir = Path(config["output_dir"]) / str(as_of.date())
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates.head(50).to_csv(run_dir / "validation_candidates_top50.csv", index=False)
    target_frame.to_csv(run_dir / "target_weights.csv", index=False)
    orders.to_csv(run_dir / "orders.csv", index=False)
    summary = {
        "as_of": str(as_of.date()),
        "next_action": "Place generated paper orders at next market open, then update positions.csv with simulated or actual fills.",
        "config": config,
        "train": f"{train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} dates)",
        "validation": f"{val_dates[0].date()} to {val_dates[-1].date()} ({len(val_dates)} dates)",
        "selected": {
            "model": asdict(best["model"]),
            "recency_scheme": asdict(best["scheme"]),
            "feature_count": len(best["cols"]),
            "trade_rule": asdict(best["trade_rule"]),
            "exposure_rule": asdict(best["exposure_rule"]),
            "validation_score": best["score"],
            "validation_sharpe": best["validation_sharpe"],
            "validation_annual_return": best["validation_annual_return"],
            "validation_max_drawdown": best["validation_max_drawdown"],
        },
        "portfolio": {
            "target_gross_exposure": float(sum(abs(v) for v in target_weights.values())),
            "target_positions": int(len(target_weights)),
            "orders": int(len(orders)),
            "buy_orders": int((orders["side"] == "BUY").sum()) if not orders.empty else 0,
            "sell_orders": int((orders["side"] == "SELL").sum()) if not orders.empty else 0,
            "estimated_buy_value": float(orders.loc[orders["side"] == "BUY", "estimated_value"].fillna(0).sum()) if not orders.empty else 0.0,
            "estimated_sell_value": float(orders.loc[orders["side"] == "SELL", "estimated_value"].fillna(0).sum()) if not orders.empty else 0.0,
        },
        "outputs": {
            "run_dir": str(run_dir),
            "orders": str(run_dir / "orders.csv"),
            "targets": str(run_dir / "target_weights.csv"),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))
    if orders.empty and not bool(config["send_empty_orders"]):
        print("No paper orders generated.")
    elif not orders.empty:
        print("\nOrders:")
        print(orders.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/paper_trading_config.example.json")
    parser.add_argument("--as-of", default="")
    parser.add_argument("--capital", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
