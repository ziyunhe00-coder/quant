from __future__ import annotations

import argparse
import json
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
from csi_fundamental_regime_exposure import ExposureRule, target_exposure
from csi_fundamental_rf_small import REPORT_BASE
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_adaptive_walkforward import block_bootstrap, make_folds
from csi_medium_term_strategy import metrics, turnover
from random_forest_a_share_strategy import clean_for_json


def execution_rule_from_name(name: str, target_vol: float) -> ExposureRule:
    if name == "trend_breadth_keep":
        return ExposureRule(2, name, "Original trend/breadth exposure guard.", "trend_breadth")
    if name.startswith("trend_vol_dd"):
        return ExposureRule(2000 + int(round(target_vol * 1000)), name, name, "trend_vol_dd", target_vol=target_vol)
    if name.startswith("trend_vol"):
        return ExposureRule(1000 + int(round(target_vol * 1000)), name, name, "trend_vol", target_vol=target_vol)
    if name.startswith("vol_"):
        return ExposureRule(int(round(target_vol * 1000)), name, name, "vol_target", target_vol=target_vol)
    raise ValueError(f"Unknown exposure rule: {name}")


def add_execution_columns(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for symbol, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        group["exec_open"] = group["open"].shift(-1)
        group["exec_amount"] = group["amount"].shift(-1)
        group["exit_open"] = group["open"].shift(-2)
        group["exec_return"] = group["exit_open"] / group["exec_open"] - 1
        group["exec_open_gap"] = group["exec_open"] / group["close"] - 1
        limit_pct = 0.20 if str(symbol).startswith(("300", "301", "688", "689")) else 0.10
        group["limit_pct"] = limit_pct
        group["buy_blocked_limit"] = group["exec_open_gap"] >= limit_pct - 0.006
        group["sell_blocked_limit"] = group["exec_open_gap"] <= -limit_pct + 0.006
        group["exec_tradable"] = group["exec_open"].notna() & group["exit_open"].notna() & (group["exec_amount"] > 0)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])


def trade_cost_bps(date: pd.Timestamp, buy_turnover: float, sell_turnover: float, args: argparse.Namespace) -> float:
    stamp_bps = args.stamp_tax_bps_after
    if date < pd.Timestamp(args.stamp_tax_cut_date):
        stamp_bps = args.stamp_tax_bps_before
    buy_bps = args.commission_bps + args.transfer_bps + args.slippage_bps
    sell_bps = args.commission_bps + args.transfer_bps + args.slippage_bps + stamp_bps
    return buy_turnover * buy_bps / 10000 + sell_turnover * sell_bps / 10000


def scale_to_exposure(weights: dict[str, float], gross_limit: float) -> dict[str, float]:
    gross = sum(abs(value) for value in weights.values())
    if gross <= 0 or gross <= gross_limit:
        return weights
    return {symbol: weight * gross_limit / gross for symbol, weight in weights.items()}


def realistic_backtest(predicted: pd.DataFrame, trade_rule, exposure_rule: ExposureRule, args: argparse.Namespace):
    rows = []
    holdings_rows = []
    weights: dict[str, float] = {}
    entry: dict[str, float] = {}
    peak: dict[str, float] = {}
    age: dict[str, int] = {}
    equity = 1.0
    equity_peak = 1.0

    for i, (date, day) in enumerate(predicted.groupby("date", sort=True)):
        day = day.copy()
        by_symbol = day.set_index("symbol")
        close_by_symbol = by_symbol["close"]

        stopped = []
        for symbol in list(weights):
            close = close_by_symbol.get(symbol, np.nan)
            if pd.isna(close):
                stopped.append(symbol)
                continue
            peak[symbol] = max(peak.get(symbol, close), float(close))
            pnl = float(close) / entry.get(symbol, float(close)) - 1
            dd_from_peak = float(close) / peak[symbol] - 1
            age[symbol] = age.get(symbol, 0) + 1
            if pnl <= trade_rule.hard_stop or dd_from_peak <= trade_rule.trailing_stop or age[symbol] >= trade_rule.max_holding_days:
                stopped.append(symbol)

        target = {symbol: weight for symbol, weight in weights.items() if symbol not in stopped}
        if i % trade_rule.rebalance_every == 0:
            target = select_diversified_weights(day, trade_rule)
        portfolio_drawdown = equity / equity_peak - 1
        desired_exposure = target_exposure(day, exposure_rule, portfolio_drawdown, trade_rule.max_gross_exposure)
        target = scale_to_exposure(target, desired_exposure)

        executed = weights.copy()
        buy_blocked = 0
        sell_blocked = 0
        lot_blocked = 0
        all_symbols = set(weights) | set(target)
        for symbol in all_symbols:
            current = weights.get(symbol, 0.0)
            desired = target.get(symbol, 0.0)
            diff = desired - current
            if abs(diff) < 1e-10:
                executed[symbol] = current
                continue
            if symbol not in by_symbol.index:
                if diff < 0:
                    sell_blocked += 1
                    executed[symbol] = current
                continue
            item = by_symbol.loc[symbol]
            tradable = bool(item.get("exec_tradable", False))
            if diff > 0:
                min_lot_weight = float(item["exec_open"]) * args.lot_size / (args.capital * equity)
                if not tradable or bool(item.get("buy_blocked_limit", False)):
                    buy_blocked += 1
                    executed[symbol] = current
                elif desired > 0 and desired < min_lot_weight:
                    lot_blocked += 1
                    executed[symbol] = current
                else:
                    executed[symbol] = desired
                    if current <= 0:
                        entry[symbol] = float(item["exec_open"])
                        peak[symbol] = float(item["exec_open"])
                        age[symbol] = 0
            elif diff < 0:
                if not tradable or bool(item.get("sell_blocked_limit", False)):
                    sell_blocked += 1
                    executed[symbol] = current
                else:
                    executed[symbol] = desired
                    if desired <= 1e-10:
                        entry.pop(symbol, None)
                        peak.pop(symbol, None)
                        age.pop(symbol, None)

        executed = {symbol: weight for symbol, weight in executed.items() if weight > 1e-10}
        buy_turnover = sum(max(0.0, executed.get(symbol, 0.0) - weights.get(symbol, 0.0)) for symbol in set(executed) | set(weights))
        sell_turnover = sum(max(0.0, weights.get(symbol, 0.0) - executed.get(symbol, 0.0)) for symbol in set(executed) | set(weights))
        day_turnover = buy_turnover + sell_turnover

        gross_return = 0.0
        stale_positions = 0
        for symbol, weight in executed.items():
            item_return = by_symbol["exec_return"].get(symbol, np.nan) if symbol in by_symbol.index else np.nan
            if pd.isna(item_return):
                stale_positions += 1
                item_return = 0.0
            contribution = weight * float(item_return)
            gross_return += contribution
            holdings_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "weight": weight,
                    "age": age.get(symbol, 0),
                    "exec_return": item_return,
                    "contribution": contribution,
                    "exec_open": by_symbol["exec_open"].get(symbol, np.nan) if symbol in by_symbol.index else np.nan,
                    "exec_open_gap": by_symbol["exec_open_gap"].get(symbol, np.nan) if symbol in by_symbol.index else np.nan,
                    "target_exposure": desired_exposure,
                }
            )

        cost = trade_cost_bps(pd.Timestamp(date), buy_turnover, sell_turnover, args)
        net_return = gross_return - cost
        equity *= 1 + net_return
        equity_peak = max(equity_peak, equity)

        if executed and 1 + net_return != 0:
            next_weights = {}
            for symbol, weight in executed.items():
                item_return = by_symbol["exec_return"].get(symbol, 0.0) if symbol in by_symbol.index else 0.0
                if pd.isna(item_return):
                    item_return = 0.0
                next_weights[symbol] = weight * (1 + float(item_return)) / (1 + net_return)
            weights = next_weights
        else:
            weights = executed

        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "equity": equity,
                "n_positions": len(weights),
                "gross_exposure": sum(abs(w) for w in weights.values()),
                "target_exposure": desired_exposure,
                "turnover": day_turnover,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "stops": len(stopped),
                "buy_blocked": buy_blocked,
                "sell_blocked": sell_blocked,
                "lot_blocked": lot_blocked,
                "stale_positions": stale_positions,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(holdings_rows)


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"execution_realism_n{args.max_symbols}_{args.report_name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    features = load_cached_features(args.max_symbols)
    features = attach_industry(features, Path(args.industry_map), args.industry_col)
    features = add_execution_columns(features)

    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    selected = pd.read_csv(args.selected_folds)
    specs = {spec.model_id: spec for spec in model_specs(features)}
    trades = {rule.rule_id: rule for rule in diversified_rules(constrained_only=True)}

    fold_rows = []
    equity_frames = []
    holding_frames = []
    for fold_id, (_, val_dates, test_dates) in enumerate(folds, start=1):
        row = selected[selected["fold_id"] == fold_id].iloc[0]
        spec = specs[int(row["selected_model_id"])]
        trade_rule = trades[int(row["selected_trade_rule_id"])]
        exp_rule = execution_rule_from_name(str(row["selected_exposure_name"]), float(row["selected_target_vol"]))
        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        test = features[features["date"].isin(test_dates)].copy()
        cols = columns_for_spec(features, spec)
        print(f"[fold {fold_id}] {spec.feature_set} trade={trade_rule.rule_id} exposure={exp_rule.name}", flush=True)
        model = fit_model(train_full, spec, cols)
        pred = predict_frame(model, test, cols)
        bt, holdings = realistic_backtest(pred, trade_rule, exp_rule, args)
        bt["fold_id"] = fold_id
        holdings["fold_id"] = fold_id
        equity_frames.append(bt)
        holding_frames.append(holdings)
        fold_rows.append({"fold_id": fold_id, **metrics(bt)})

    folds_df = pd.DataFrame(fold_rows)
    equity = pd.concat(equity_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)
    equity["combined_equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    combined = metrics(equity.assign(equity=equity["combined_equity"]))
    returns = equity["net_return"].fillna(0)
    diagnostics = {
        "buy_blocked_days": int((equity["buy_blocked"] > 0).sum()),
        "sell_blocked_days": int((equity["sell_blocked"] > 0).sum()),
        "lot_blocked_days": int((equity["lot_blocked"] > 0).sum()),
        "total_buy_blocked_orders": int(equity["buy_blocked"].sum()),
        "total_sell_blocked_orders": int(equity["sell_blocked"].sum()),
        "total_lot_blocked_orders": int(equity["lot_blocked"].sum()),
        "avg_daily_buy_turnover": float(equity["buy_turnover"].mean()),
        "avg_daily_sell_turnover": float(equity["sell_turnover"].mean()),
    }
    summary = {
        "objective": "Execution-realistic A-share backtest using next-open execution, limit/suspension blocks, lot-size blocks, and tax/fee/slippage assumptions.",
        "assumptions": {
            "signal_timing": "signals generated after close date t; orders attempt next open",
            "return_timing": "executed positions earn next-open to following-open return",
            "limit_rule": "10% for main-board symbols, 20% for 300/301/688/689 symbols; ST 5% not modeled because ST flag is unavailable",
            "fees": {
                "commission_bps_each_side": args.commission_bps,
                "transfer_bps_each_side": args.transfer_bps,
                "slippage_bps_each_side": args.slippage_bps,
                "stamp_tax_bps_sell_before_cut": args.stamp_tax_bps_before,
                "stamp_tax_bps_sell_after_cut": args.stamp_tax_bps_after,
                "stamp_tax_cut_date": args.stamp_tax_cut_date,
            },
            "capital": args.capital,
            "lot_size": args.lot_size,
        },
        "combined": clean_for_json(combined),
        "folds": clean_for_json(folds_df.to_dict(orient="records")),
        "bootstrap_sharpe": block_bootstrap(returns, args.bootstrap, args.block_len, args.seed),
        "diagnostics": diagnostics,
    }
    folds_df.to_csv(report_dir / "execution_folds.csv", index=False)
    equity.to_csv(report_dir / "execution_equity.csv", index=False)
    holdings.to_csv(report_dir / "execution_holdings.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nExecution-realistic folds:")
    print(folds_df.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=200)
    parser.add_argument("--report-name", default="v1_nested_sharpe")
    parser.add_argument("--selected-folds", default="reports/csi_fundamental_rf_small/true_sharpe_n200_v1_sharpe_nested/nested_folds.csv")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=194)
    parser.add_argument("--test-len", type=int, default=194)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--industry-map", default="data/processed/csi1000_industry_map_200.csv")
    parser.add_argument("--industry-col", default="industry_gate")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--commission-bps", type=float, default=2.5)
    parser.add_argument("--transfer-bps", type=float, default=0.1)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stamp-tax-bps-before", type=float, default=10.0)
    parser.add_argument("--stamp-tax-bps-after", type=float, default=5.0)
    parser.add_argument("--stamp-tax-cut-date", default="2023-08-28")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
