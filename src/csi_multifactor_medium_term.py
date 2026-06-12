from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from csi_medium_term_strategy import load_features
from random_forest_a_share_strategy import REPORT_DIR, clean_for_json


REPORT_DIR_MULTI = REPORT_DIR / "csi_multifactor_medium_term"


@dataclass(frozen=True)
class FactorRule:
    rule_id: int
    change: str
    top_n: int
    rebalance_every: int
    market_filter: str
    stock_filter: str
    weighting: str
    score_quantile: float
    max_weight: float
    hard_stop: float
    trailing_stop: float
    atr_stop_mult: float
    max_holding_days: int
    transaction_cost_bps: float = 14.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def enrich_multifactor_features(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        prev_close = group["close"].shift(1)
        tr = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - prev_close).abs(),
                (group["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        group["atr_pct_20"] = tr.rolling(20).mean() / group["close"]
        group["ret_120"] = group["close"].pct_change(120)
        group["ma_gap_120"] = group["close"] / group["close"].rolling(120).mean() - 1
        high_60 = group["high"].rolling(60).max()
        group["high_gap_60"] = group["close"] / high_60 - 1
        pieces.append(group)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    out["market_ret_120"] = out.groupby("date")["ret_120"].transform("mean")
    out["rel_ret_120"] = out["ret_120"] - out["market_ret_120"]
    out["market_breadth_60"] = out.groupby("date")["ret_60"].transform(lambda x: float((x > 0).mean()))
    out["market_breadth_120"] = out.groupby("date")["ret_120"].transform(lambda x: float((x > 0).mean()))
    return out.replace([np.inf, -np.inf], np.nan)


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[
        frame["ret_1"].between(-0.22, 0.22)
        & frame["fwd_return_1"].between(-0.22, 0.22)
        & frame["atr_pct_20"].between(0.005, 0.20)
    ].copy()
    return out


def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending)


def add_factor_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = []
    for _, day in frame.groupby("date", sort=True):
        day = day.copy()
        # Higher is better for momentum/relative strength/liquidity. Lower is better for volatility.
        day["rank_ret_60"] = percentile_rank(day["ret_60"])
        day["rank_ret_120"] = percentile_rank(day["ret_120"])
        day["rank_rel_60"] = percentile_rank(day["rel_ret_60"])
        day["rank_rel_120"] = percentile_rank(day["rel_ret_120"])
        day["rank_low_vol"] = percentile_rank(day["vol_20"], ascending=False)
        day["rank_liquidity"] = percentile_rank(day["amount"])
        day["rank_trend"] = percentile_rank(day["ma_gap_120"])
        day["rank_not_overheat"] = percentile_rank(-(day["rsi_14"] - 58).abs())
        day["factor_score"] = (
            0.22 * day["rank_ret_60"]
            + 0.18 * day["rank_ret_120"]
            + 0.18 * day["rank_rel_60"]
            + 0.12 * day["rank_rel_120"]
            + 0.12 * day["rank_low_vol"]
            + 0.08 * day["rank_liquidity"]
            + 0.06 * day["rank_trend"]
            + 0.04 * day["rank_not_overheat"]
        )
        scored.append(day)
    return pd.concat(scored, ignore_index=True).sort_values(["date", "symbol"])


def rules() -> list[FactorRule]:
    return [
        FactorRule(1, "基线多因子：Top30，20日调仓，放宽15/20止损。", 30, 20, "trend_ok", "core", "inverse_vol", 0.80, 0.06, -0.15, -0.20, 3.5, 120),
        FactorRule(2, "更分散：Top50，20日调仓，ATR止损为主。", 50, 20, "trend_ok", "core", "inverse_vol", 0.75, 0.04, -0.18, -0.24, 4.0, 160),
        FactorRule(3, "更集中：Top20，强市场才交易。", 20, 20, "strong", "core", "equal", 0.85, 0.08, -0.14, -0.20, 3.5, 100),
        FactorRule(4, "低波质量：Top40，弱市更防守。", 40, 20, "defensive", "lowvol_quality", "inverse_vol", 0.75, 0.05, -0.14, -0.20, 3.2, 140),
        FactorRule(5, "长期趋势：Top30，40日调仓。", 30, 40, "trend_ok", "long_trend", "inverse_vol", 0.80, 0.06, -0.18, -0.25, 4.0, 180),
        FactorRule(6, "小资金进攻但放宽止损：Top12。", 12, 20, "strong", "core", "equal", 0.88, 0.12, -0.16, -0.22, 3.8, 100),
    ]


def market_allows(day: pd.DataFrame, rule: FactorRule) -> bool:
    first = day.iloc[0]
    if rule.market_filter == "trend_ok":
        return first["market_breadth_60"] > 0.48 and (first["market_ret_60"] > -0.04 or first["market_ret_120"] > 0)
    if rule.market_filter == "strong":
        return first["market_breadth_60"] > 0.55 and first["market_ret_60"] > 0 and first["market_ret_120"] > -0.03
    if rule.market_filter == "defensive":
        return first["market_breadth_60"] > 0.42 and first["market_ret_120"] > -0.08
    raise ValueError(rule.market_filter)


def tradable_mask(day: pd.DataFrame) -> pd.Series:
    symbols = day["symbol"].astype(str).str.zfill(6)
    wide = symbols.str.startswith(("300", "301", "688", "689"))
    limit = pd.Series(np.where(wide, 0.18, 0.092), index=day.index)
    return day["ret_1"] < limit


def filter_stocks(day: pd.DataFrame, rule: FactorRule) -> pd.DataFrame:
    day = day[tradable_mask(day)].copy()
    amount_cut = day["amount"].quantile(0.35)
    if rule.stock_filter == "core":
        return day[
            (day["amount"] >= amount_cut)
            & (day["ret_120"] > -0.05)
            & (day["ma_gap_120"] > -0.08)
            & (day["ret_1"] < 0.07)
            & (day["rsi_14"].between(35, 78))
        ]
    if rule.stock_filter == "lowvol_quality":
        return day[
            (day["amount"] >= amount_cut)
            & (day["ret_60"] > -0.05)
            & (day["ma_gap_60"] > -0.05)
            & (day["vol_20"] < day["vol_20"].quantile(0.60))
            & (day["rsi_14"].between(35, 72))
        ]
    if rule.stock_filter == "long_trend":
        return day[
            (day["amount"] >= amount_cut)
            & (day["ret_120"] > 0)
            & (day["ma_gap_120"] > -0.03)
            & (day["high_gap_60"] > -0.20)
            & (day["rsi_14"].between(35, 80))
        ]
    raise ValueError(rule.stock_filter)


def cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights / weights.sum()
    capped = weights.copy()
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


def select_weights(day: pd.DataFrame, rule: FactorRule) -> dict[str, float]:
    if not market_allows(day, rule):
        return {}
    day = filter_stocks(day.dropna(subset=["factor_score", "vol_20", "fwd_return_1", "close", "atr_pct_20"]), rule)
    if day.empty:
        return {}
    threshold = day["factor_score"].quantile(rule.score_quantile)
    selected = day[day["factor_score"] >= threshold].sort_values("factor_score", ascending=False).head(rule.top_n)
    if selected.empty:
        return {}
    if rule.weighting == "equal":
        raw = pd.Series(1.0, index=selected["symbol"])
    elif rule.weighting == "inverse_vol":
        raw = 1 / selected.set_index("symbol")["vol_20"].replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        raise ValueError(rule.weighting)
    return cap_weights(raw, rule.max_weight).to_dict()


def turnover(old: dict[str, float], new: dict[str, float]) -> float:
    return float(sum(abs(old.get(s, 0.0) - new.get(s, 0.0)) for s in set(old) | set(new)))


def backtest(frame: pd.DataFrame, rule: FactorRule) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    holdings_rows = []
    weights: dict[str, float] = {}
    entry: dict[str, float] = {}
    peak: dict[str, float] = {}
    entry_atr: dict[str, float] = {}
    age: dict[str, int] = {}
    equity = 1.0
    for i, (date, day) in enumerate(frame.groupby("date", sort=True)):
        close_by_symbol = day.set_index("symbol")["close"]
        return_by_symbol = day.set_index("symbol")["fwd_return_1"]
        atr_by_symbol = day.set_index("symbol")["atr_pct_20"]

        stopped = []
        for symbol in list(weights):
            close = close_by_symbol.get(symbol, np.nan)
            if pd.isna(close):
                stopped.append(symbol)
                continue
            peak[symbol] = max(peak.get(symbol, float(close)), float(close))
            pnl = float(close) / entry[symbol] - 1
            peak_dd = float(close) / peak[symbol] - 1
            atr_stop = -rule.atr_stop_mult * entry_atr.get(symbol, 0.04)
            effective_hard = min(rule.hard_stop, atr_stop)
            age[symbol] = age.get(symbol, 0) + 1
            if pnl <= effective_hard or peak_dd <= rule.trailing_stop or age[symbol] >= rule.max_holding_days:
                stopped.append(symbol)
        for symbol in stopped:
            weights.pop(symbol, None)
            entry.pop(symbol, None)
            peak.pop(symbol, None)
            entry_atr.pop(symbol, None)
            age.pop(symbol, None)

        target = weights.copy()
        if i % rule.rebalance_every == 0:
            target = select_weights(day, rule)
            for symbol in target:
                close = close_by_symbol.get(symbol, np.nan)
                atr = atr_by_symbol.get(symbol, np.nan)
                if symbol not in entry and pd.notna(close):
                    entry[symbol] = float(close)
                    peak[symbol] = float(close)
                    entry_atr[symbol] = float(atr) if pd.notna(atr) else 0.04
                    age[symbol] = 0
            for symbol in list(entry):
                if symbol not in target:
                    entry.pop(symbol, None)
                    peak.pop(symbol, None)
                    entry_atr.pop(symbol, None)
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
        return {}
    returns = bt["net_return"].fillna(0)
    std = returns.std(ddof=0)
    sharpe = 0.0 if std == 0 else float(returns.mean() / std * np.sqrt(252))
    drawdown = bt["equity"] / bt["equity"].cummax() - 1
    active = returns[returns != 0]
    return {
        "sharpe": sharpe,
        "annual_return": float(bt["equity"].iloc[-1] ** (252 / max(1, len(bt))) - 1),
        "annual_vol": float(std * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(bt["equity"].iloc[-1] - 1),
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "avg_positions": float(bt["n_positions"].mean()),
        "avg_exposure": float(bt["gross_exposure"].mean()),
        "avg_turnover": float(bt["turnover"].mean()),
        "stop_days": int((bt["stops"] > 0).sum()),
    }


def split_dates(frame: pd.DataFrame):
    dates = pd.Index(sorted(frame.dropna(subset=["ret_120", "fwd_return_1"])["date"].unique()))
    n_train = int(len(dates) * 0.70)
    n_val = int(len(dates) * 0.10)
    return dates[:n_train], dates[n_train : n_train + n_val], dates[n_train + n_val :]


def evaluate(frame: pd.DataFrame, report_dir: Path):
    train_dates, val_dates, live_dates = split_dates(frame)
    val = frame[frame["date"].isin(val_dates)].copy()
    live = frame[frame["date"].isin(live_dates)].copy()
    rows = []
    artifacts = {}
    for rule in rules():
        val_bt, val_h = backtest(val, rule)
        live_bt, live_h = backtest(live, rule)
        val_m = metrics(val_bt)
        live_m = metrics(live_bt)
        score = val_m["sharpe"] + 0.25 * val_m["win_rate"] - max(0.0, abs(val_m["max_drawdown"]) - 0.20)
        row = {
            "rule_id": rule.rule_id,
            "change": rule.change,
            "validation_score": score,
            **{f"validation_{k}": v for k, v in val_m.items()},
            **{f"paper_live_{k}": v for k, v in live_m.items()},
            **{f"rule_{k}": v for k, v in asdict(rule).items()},
        }
        rows.append(row)
        artifacts[rule.rule_id] = (val_bt, val_h, live_bt, live_h)
    candidates = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    candidates.to_csv(report_dir / "validation_candidates.csv", index=False)
    return candidates, artifacts, train_dates, val_dates, live_dates


def make_folds(dates: pd.Index, train_min: int = 760, test_len: int = 126) -> list[tuple[pd.Index, pd.Index]]:
    folds = []
    start = train_min
    while start + test_len <= len(dates):
        folds.append((dates[:start], dates[start : start + test_len]))
        start += test_len
    return folds


def robust_check(frame: pd.DataFrame, rule: FactorRule, report_dir: Path) -> pd.DataFrame:
    dates = pd.Index(sorted(frame.dropna(subset=["ret_120", "fwd_return_1"])["date"].unique()))
    rows = []
    equities = []
    for fold_id, (_, test_dates) in enumerate(make_folds(dates), start=1):
        test = frame[frame["date"].isin(test_dates)].copy()
        bt, _ = backtest(test, rule)
        bt["fold_id"] = fold_id
        equities.append(bt)
        rows.append(
            {
                "fold_id": fold_id,
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                **metrics(bt),
            }
        )
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(report_dir / f"rule_{rule.rule_id}_walkforward_folds.csv", index=False)
    if equities:
        pd.concat(equities, ignore_index=True).to_csv(report_dir / f"rule_{rule.rule_id}_walkforward_equity.csv", index=False)
    return fold_df


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_DIR_MULTI / args.market
    ensure_dir(report_dir)
    frame = add_factor_scores(clean_features(enrich_multifactor_features(load_features(args.market))))
    candidates, artifacts, train_dates, val_dates, live_dates = evaluate(frame, report_dir)
    top_rows = []
    rule_by_id = {rule.rule_id: rule for rule in rules()}
    for _, row in candidates.head(args.finalists).iterrows():
        rule = rule_by_id[int(row["rule_id"])]
        _, _, live_bt, live_h = artifacts[rule.rule_id]
        live_bt.to_csv(report_dir / f"candidate_{rule.rule_id}_equity.csv", index=False)
        live_h.to_csv(report_dir / f"candidate_{rule.rule_id}_holdings.csv", index=False)
        fold_df = robust_check(frame, rule, report_dir)
        top_rows.append(
            {
                **row.to_dict(),
                "wf_avg_sharpe": float(fold_df["sharpe"].mean()),
                "wf_median_sharpe": float(fold_df["sharpe"].median()),
                "wf_min_sharpe": float(fold_df["sharpe"].min()),
                "wf_positive_folds": int((fold_df["sharpe"] > 0).sum()),
            }
        )
    top = pd.DataFrame(top_rows).sort_values(["wf_median_sharpe", "paper_live_sharpe"], ascending=False)
    top.to_csv(report_dir / "top_final_checks.csv", index=False)
    best = top.iloc[0].to_dict()
    summary = {
        "objective": "Transparent multifactor medium-term strategy with looser volatility-aware stops.",
        "market": args.market,
        "dataset": {
            "symbols": int(frame["symbol"].nunique()),
            "rows": int(len(frame)),
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
        },
        "best": clean_for_json(best),
        "top": clean_for_json(top.head(args.finalists).to_dict(orient="records")),
    }
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(top[["rule_id", "validation_sharpe", "paper_live_sharpe", "paper_live_annual_return", "paper_live_max_drawdown", "wf_avg_sharpe", "wf_median_sharpe", "wf_min_sharpe", "wf_positive_folds"]].to_string(index=False))
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["csi1000", "csi500", "combined"], default="csi1000")
    parser.add_argument("--finalists", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
