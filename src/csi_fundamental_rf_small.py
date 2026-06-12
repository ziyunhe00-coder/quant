from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from csi_medium_term_adaptive_walkforward import block_bootstrap
from csi_medium_term_strategy import (
    MediumRule,
    backtest,
    clean_anomalies,
    load_features,
    metrics,
    rules,
)
from random_forest_a_share_strategy import PROCESSED_DIR, REPORT_DIR, clean_for_json, split_dates


RAW_FUND_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_fundamentals_em"
FINANCIAL_DIR = RAW_FUND_DIR / "financial"
VALUATION_DIR = RAW_FUND_DIR / "valuation"
REPORT_BASE = REPORT_DIR / "csi_fundamental_rf_small"


PRICE_FEATURES = [
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
    "trend_quality_20",
    "amount_ratio_5_20",
    "high_gap_20",
]

VALUATION_RENAME = {
    "总市值": "val_total_mv",
    "流通市值": "val_float_mv",
    "总股本": "val_total_shares",
    "流通股本": "val_float_shares",
    "PE(TTM)": "val_pe_ttm",
    "PE(静)": "val_pe_static",
    "市净率": "val_pb",
    "PEG值": "val_peg",
    "市现率": "val_pcf",
    "市销率": "val_ps",
}

FINANCIAL_RENAME = {
    "EPSJB": "fin_eps_basic",
    "EPSKCJB": "fin_eps_deducted",
    "BPS": "fin_bps",
    "MGJYXJJE": "fin_ocf_per_share",
    "TOTALOPERATEREVE": "fin_revenue",
    "MLR": "fin_gross_profit",
    "PARENTNETPROFIT": "fin_parent_profit",
    "KCFJCXSYJLR": "fin_deduct_profit",
    "TOTALOPERATEREVETZ": "fin_revenue_yoy",
    "PARENTNETPROFITTZ": "fin_parent_profit_yoy",
    "KCFJCXSYJLRTZ": "fin_deduct_profit_yoy",
    "ROEJQ": "fin_roe",
    "ROEKCJQ": "fin_roe_deducted",
    "ZZCJLL": "fin_roa",
    "XSJLL": "fin_net_margin",
    "XSMLL": "fin_gross_margin",
    "JYXJLYYSR": "fin_ocf_to_revenue",
    "LD": "fin_current_ratio",
    "SD": "fin_quick_ratio",
    "ZCFZL": "fin_debt_to_assets",
    "ROIC": "fin_roic",
    "DJD_TOI_YOY": "fin_single_q_revenue_yoy",
    "DJD_DPNP_YOY": "fin_single_q_profit_yoy",
    "DJD_DEDUCTDPNP_YOY": "fin_single_q_deduct_profit_yoy",
}

RANK_BASE_FEATURES = [
    "val_earnings_yield",
    "val_book_to_price",
    "val_sales_to_price",
    "val_log_total_mv",
    "val_log_float_mv",
    "fin_roe",
    "fin_roe_deducted",
    "fin_roa",
    "fin_roic",
    "fin_net_margin",
    "fin_gross_margin",
    "fin_revenue_yoy",
    "fin_parent_profit_yoy",
    "fin_deduct_profit_yoy",
    "fin_single_q_revenue_yoy",
    "fin_single_q_profit_yoy",
    "fin_debt_to_assets",
    "fin_ocf_to_revenue",
    "fin_days_since_notice",
]


@dataclass(frozen=True)
class FundamentalModel:
    model_id: int
    feature_set: str
    change: str
    target: str = "target_rank_40"
    n_estimators: int = 260
    max_depth: int | None = 6
    min_samples_leaf: int = 70
    max_features: str | float = "sqrt"
    max_samples: float = 0.45
    random_state: int = 42


def ensure_dirs() -> None:
    for path in (FINANCIAL_DIR, VALUATION_DIR, REPORT_BASE):
        path.mkdir(parents=True, exist_ok=True)


def exchange_symbol(symbol: str) -> str:
    suffix = "SH" if str(symbol).startswith(("6", "688", "689")) else "SZ"
    return f"{str(symbol).zfill(6)}.{suffix}"


def select_liquid_symbols(features: pd.DataFrame, max_symbols: int) -> list[str]:
    last_date = features["date"].max()
    start = last_date - pd.Timedelta(days=365)
    recent = features[(features["date"] >= start) & (features["amount"] > 0)].copy()
    liquidity = (
        recent.groupby("symbol", as_index=False)
        .agg(median_amount=("amount", "median"), rows=("date", "count"))
        .query("rows >= 180")
        .sort_values("median_amount", ascending=False)
    )
    return liquidity["symbol"].astype(str).str.zfill(6).head(max_symbols).tolist()


def fetch_or_load_financial(symbol: str, refresh: bool, pause: float) -> pd.DataFrame | None:
    path = FINANCIAL_DIR / f"{symbol}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    import akshare as ak

    try:
        raw = ak.stock_financial_analysis_indicator_em(symbol=exchange_symbol(symbol), indicator="按报告期")
        if raw.empty:
            return None
        raw.to_csv(path, index=False)
        time.sleep(pause)
        return raw
    except Exception as exc:  # noqa: BLE001
        print(f"[financial] {symbol} failed: {exc}", flush=True)
        time.sleep(pause)
        return None


def fetch_or_load_valuation(symbol: str, refresh: bool, pause: float) -> pd.DataFrame | None:
    path = VALUATION_DIR / f"{symbol}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    import akshare as ak

    try:
        raw = ak.stock_value_em(symbol=str(symbol).zfill(6))
        if raw.empty:
            return None
        raw.to_csv(path, index=False)
        time.sleep(pause)
        return raw
    except Exception as exc:  # noqa: BLE001
        print(f"[valuation] {symbol} failed: {exc}", flush=True)
        time.sleep(pause)
        return None


def slug_column(name: str, fallback: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_").lower()
    return f"fin_{slug}" if slug else fallback


def normalize_valuation(symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "数据日期" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["数据日期"], errors="coerce")
    keep = ["date"]
    for source, target in VALUATION_RENAME.items():
        if source in df.columns:
            df[target] = pd.to_numeric(df[source], errors="coerce")
            keep.append(target)
    out = df[keep].dropna(subset=["date"]).sort_values("date").copy()
    feature_cols = [col for col in out.columns if col != "date"]
    out[feature_cols] = out[feature_cols].shift(1)
    out["symbol"] = symbol
    return out.drop_duplicates(["symbol", "date"], keep="last")


def normalize_financial(symbol: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    if "NOTICE_DATE" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    df["notice_date"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce")
    df["report_date"] = pd.to_datetime(df.get("REPORT_DATE"), errors="coerce")
    df["effective_date"] = df["notice_date"] + pd.Timedelta(days=1)

    used = []
    out = df[["effective_date", "notice_date", "report_date"]].copy()
    excluded = {
        "NOTICE_DATE",
        "REPORT_DATE",
        "SECURITY_CODE",
        "SECURITY_NAME_ABBR",
        "REPORT_YEAR",
        "notice_date",
        "report_date",
        "effective_date",
    }
    for i, col in enumerate(df.columns):
        if col in excluded:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() < 3:
            continue
        target = FINANCIAL_RENAME.get(col, slug_column(col, f"fin_col_{i}"))
        if target in out.columns:
            target = f"{target}_{i}"
        out[target] = numeric
        used.append({"source_column": col, "feature_column": target})

    out["symbol"] = symbol
    out = out.dropna(subset=["effective_date"]).sort_values("effective_date")
    dictionary = pd.DataFrame(used)
    return out.drop_duplicates(["symbol", "effective_date"], keep="last"), dictionary


def merge_symbol_fundamentals(
    symbol_frame: pd.DataFrame,
    valuation: pd.DataFrame,
    financial: pd.DataFrame,
) -> pd.DataFrame:
    symbol_frame = symbol_frame.sort_values("date").copy()
    if not valuation.empty:
        symbol_frame = symbol_frame.merge(valuation.drop(columns=["symbol"]), on="date", how="left")
    if not financial.empty:
        financial = financial.sort_values("effective_date")
        symbol_frame = pd.merge_asof(
            symbol_frame.sort_values("date"),
            financial.drop(columns=["symbol"]).sort_values("effective_date"),
            left_on="date",
            right_on="effective_date",
            direction="backward",
        )
        symbol_frame["fin_days_since_notice"] = (
            symbol_frame["date"] - symbol_frame["notice_date"]
        ).dt.days.astype("float")
    return symbol_frame


def add_targets(features: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("date").copy()
        group["fwd_return_20"] = group["close"].pct_change(20).shift(-20)
        group["fwd_return_40"] = group["close"].pct_change(40).shift(-40)
        pieces.append(group)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    out["target_rank_20"] = out.groupby("date")["fwd_return_20"].rank(pct=True)
    out["target_rank_40"] = out.groupby("date")["fwd_return_40"].rank(pct=True)
    out["target_excess_20"] = out["fwd_return_20"] - out.groupby("date")["fwd_return_20"].transform("mean")
    return out.replace([np.inf, -np.inf], np.nan)


def add_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for source in ["val_total_mv", "val_float_mv"]:
        if source in out.columns:
            out[source.replace("val_", "val_log_")] = np.log1p(out[source].where(out[source] > 0))
    if "val_pe_ttm" in out.columns:
        out["val_earnings_yield"] = np.where(out["val_pe_ttm"] > 0, 1 / out["val_pe_ttm"], np.nan)
    if "val_pb" in out.columns:
        out["val_book_to_price"] = np.where(out["val_pb"] > 0, 1 / out["val_pb"], np.nan)
    if "val_ps" in out.columns:
        out["val_sales_to_price"] = np.where(out["val_ps"] > 0, 1 / out["val_ps"], np.nan)

    rank_cols = [col for col in RANK_BASE_FEATURES if col in out.columns]
    for col in rank_cols:
        out[f"{col}_rank"] = out.groupby("date")[col].rank(pct=True)
    return out.replace([np.inf, -np.inf], np.nan)


def build_fundamental_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = clean_anomalies(load_features("csi1000"))
    symbols = select_liquid_symbols(base, args.max_symbols)
    if args.symbols:
        symbols = [item.strip().zfill(6) for item in args.symbols.split(",") if item.strip()]
    print(f"[universe] {len(symbols)} symbols: {','.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}", flush=True)

    rows = []
    dictionaries = []
    failures = []
    for i, symbol in enumerate(symbols, start=1):
        print(f"[fundamental] {i}/{len(symbols)} {symbol}", flush=True)
        val_raw = fetch_or_load_valuation(symbol, args.refresh_fundamentals, args.pause)
        fin_raw = fetch_or_load_financial(symbol, args.refresh_fundamentals, args.pause)
        valuation = normalize_valuation(symbol, val_raw) if val_raw is not None else pd.DataFrame()
        financial, dictionary = normalize_financial(symbol, fin_raw) if fin_raw is not None else (pd.DataFrame(), pd.DataFrame())
        if valuation.empty and financial.empty:
            failures.append({"symbol": symbol, "reason": "missing valuation and financial data"})
        if not dictionary.empty:
            dictionary["symbol"] = symbol
            dictionaries.append(dictionary)
        symbol_frame = base[base["symbol"] == symbol].copy()
        rows.append(merge_symbol_fundamentals(symbol_frame, valuation, financial))

    feature_dictionary = pd.concat(dictionaries, ignore_index=True) if dictionaries else pd.DataFrame()
    if failures:
        pd.DataFrame(failures).to_csv(REPORT_BASE / "data_failures.csv", index=False)
    dataset = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    dataset = add_derived_features(add_targets(dataset))
    dataset.to_parquet(PROCESSED_DIR / f"csi1000_fundamental_small_{len(symbols)}.parquet", index=False)
    feature_dictionary.to_csv(REPORT_BASE / "feature_dictionary.csv", index=False)
    return dataset, feature_dictionary


def model_configs() -> list[FundamentalModel]:
    return [
        FundamentalModel(1, "price", "价格量价基线：只用之前中线价量/市场特征。", min_samples_leaf=60),
        FundamentalModel(2, "price_value", "加入每日估值：PE/PB/PS/PCF/市值及反向估值。", min_samples_leaf=60),
        FundamentalModel(3, "price_value_fin", "加入公告日对齐后的财报质量、成长、现金流、杠杆等全部可转数值字段。"),
        FundamentalModel(4, "value_fin", "去掉大部分价量，只看估值+基本面是否有独立选股能力。", min_samples_leaf=50),
        FundamentalModel(
            5,
            "price_value_fin",
            "更长叶子、更浅树：降低小样本基本面过拟合。",
            max_depth=5,
            min_samples_leaf=110,
            max_samples=0.40,
        ),
    ]


def candidate_rules() -> list[MediumRule]:
    base = rules()
    return [
        base[1],
        base[2],
        base[4],
        MediumRule(201, 1, "Top8质量趋势，20日调仓，12/18宽止损。", 8, 20, "trend_ok", "quality_trend", "inverse_vol", 0.80, 0.15, -0.12, -0.18, 90),
        MediumRule(202, 1, "Top12相对强度，20日调仓，12/18宽止损。", 12, 20, "trend_ok", "relative_strength", "equal", 0.80, 0.12, -0.12, -0.18, 90),
        MediumRule(203, 1, "Top15质量趋势，40日调仓，15/22宽止损。", 15, 40, "defensive", "quality_trend", "inverse_vol", 0.75, 0.10, -0.15, -0.22, 120),
    ]


def feature_columns(features: pd.DataFrame, config: FundamentalModel) -> list[str]:
    value_cols = [col for col in features.columns if col.startswith("val_")]
    fin_cols = [col for col in features.columns if col.startswith("fin_")]
    if config.feature_set == "price":
        cols = PRICE_FEATURES
    elif config.feature_set == "price_value":
        cols = PRICE_FEATURES + value_cols
    elif config.feature_set == "price_value_fin":
        cols = PRICE_FEATURES + value_cols + fin_cols
    elif config.feature_set == "value_fin":
        cols = value_cols + fin_cols + ["market_ret_20", "market_ret_60", "market_breadth_20"]
    else:
        raise ValueError(config.feature_set)
    return [col for col in dict.fromkeys(cols) if col in features.columns]


def fit_rf(train: pd.DataFrame, config: FundamentalModel, cols: list[str]) -> Pipeline:
    sample = train.dropna(subset=[config.target]).copy()
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
    pipe.fit(sample[cols], sample[config.target])
    return pipe


def predict_rf(model: Pipeline, frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(out[cols])
    return out


def score_validation(m: dict[str, float]) -> float:
    exposure_penalty = max(0.0, 0.20 - m.get("avg_exposure", 0.0)) * 2.0
    position_penalty = max(0.0, 3.0 - m.get("avg_positions", 0.0)) * 0.2
    dd_penalty = max(0.0, abs(m.get("max_drawdown", 0.0)) - 0.20) * 1.5
    return m["sharpe"] + 0.20 * m.get("win_rate", 0.0) - exposure_penalty - position_penalty - dd_penalty


def evaluate_80_10_10(features: pd.DataFrame, report_dir: Path) -> tuple[pd.DataFrame, dict]:
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    train_dates, val_dates, live_dates = split_dates(eligible)
    train = features[features["date"].isin(train_dates)].copy()
    val = features[features["date"].isin(val_dates)].copy()
    live = features[features["date"].isin(live_dates)].copy()
    rows = []
    equity_frames = {}
    holding_frames = {}

    for config in model_configs():
        cols = feature_columns(features, config)
        print(f"[model {config.model_id}] {config.feature_set}, features={len(cols)}", flush=True)
        model = fit_rf(train, config, cols)
        val_pred = predict_rf(model, val, cols)
        live_pred_train_only = predict_rf(model, live, cols)
        for rule in candidate_rules():
            val_bt, _ = backtest(val_pred, rule)
            live_bt, _ = backtest(live_pred_train_only, rule)
            val_m = metrics(val_bt)
            live_m = metrics(live_bt)
            rows.append(
                {
                    "model_id": config.model_id,
                    "rule_id": rule.rule_id,
                    "feature_set": config.feature_set,
                    "feature_count": len(cols),
                    "model_change": config.change,
                    "rule_change": rule.change,
                    "validation_score": score_validation(val_m),
                    **{f"validation_{k}": v for k, v in val_m.items()},
                    **{f"paper_live_train_only_{k}": v for k, v in live_m.items()},
                    **{f"model_{k}": v for k, v in asdict(config).items()},
                    **{f"rule_{k}": v for k, v in asdict(rule).items()},
                }
            )

    candidates = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
    candidates.to_csv(report_dir / "validation_candidates.csv", index=False)
    best = candidates.iloc[0]
    best_config = next(item for item in model_configs() if item.model_id == int(best["model_id"]))
    best_rule = next(item for item in candidate_rules() if item.rule_id == int(best["rule_id"]))
    best_cols = feature_columns(features, best_config)
    final_model = fit_rf(features[features["date"].isin(list(train_dates) + list(val_dates))], best_config, best_cols)
    final_pred = predict_rf(final_model, live, best_cols)
    final_bt, final_holdings = backtest(final_pred, best_rule)
    final_bt.to_csv(report_dir / "best_final_equity.csv", index=False)
    final_holdings.to_csv(report_dir / "best_final_holdings.csv", index=False)
    equity_frames["best_final"] = final_bt
    holding_frames["best_final"] = final_holdings

    summary = {
        "split": {
            "train": f"{train_dates.min().date()} to {train_dates.max().date()} ({len(train_dates)} dates)",
            "validation": f"{val_dates.min().date()} to {val_dates.max().date()} ({len(val_dates)} dates)",
            "paper_live": f"{live_dates.min().date()} to {live_dates.max().date()} ({len(live_dates)} dates)",
        },
        "best_validation_row": clean_for_json(best.to_dict()),
        "best_model": asdict(best_config),
        "best_rule": asdict(best_rule),
        "best_final_metrics": metrics(final_bt),
        "bootstrap_final_sharpe": block_bootstrap(final_bt["net_return"].fillna(0), 300, 20, 42),
    }
    return candidates, summary


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    report_dir = REPORT_BASE / f"n{args.max_symbols}"
    report_dir.mkdir(parents=True, exist_ok=True)
    features, feature_dictionary = build_fundamental_dataset(args)
    features = features[features["amount"] > 0].copy()
    candidates, summary = evaluate_80_10_10(features, report_dir)
    summary.update(
        {
            "objective": "Small-sample CSI1000 random forest with point-in-time fundamentals and lagged valuation.",
            "dataset": {
                "symbols": int(features["symbol"].nunique()),
                "rows": int(len(features)),
                "start": str(features["date"].min().date()),
                "end": str(features["date"].max().date()),
                "financial_feature_dictionary_rows": int(len(feature_dictionary)),
            },
            "data_controls": {
                "financial_effective_date": "NOTICE_DATE + 1 calendar day",
                "valuation": "daily valuation fields shifted by one row per symbol",
                "universe": "top recent-liquidity CSI1000 symbols unless --symbols is provided",
                "transaction_cost_bps": 14,
            },
            "top_candidates": clean_for_json(candidates.head(10).to_dict(orient="records")),
        }
    )
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        candidates[
            [
                "model_id",
                "rule_id",
                "feature_set",
                "feature_count",
                "validation_sharpe",
                "paper_live_train_only_sharpe",
                "paper_live_train_only_annual_return",
                "paper_live_train_only_max_drawdown",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )
    print("\nFinal paper-live retrained on train+validation:")
    print(json.dumps(clean_for_json(summary["best_final_metrics"]), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=40)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--refresh-fundamentals", action="store_true")
    parser.add_argument("--pause", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
