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

from csi_fundamental_improvement_analysis import (
    attach_industry,
    columns_for_spec,
    diversified_rules,
    model_specs,
    predict_frame,
)
from csi_fundamental_regime_exposure import combined_metrics, dynamic_backtest, regime_score
from csi_fundamental_rf_small import REPORT_BASE
from csi_fundamental_true_sharpe_validation import exposure_candidates
from csi_fundamental_walkforward import load_cached_features
from csi_medium_term_adaptive_walkforward import block_bootstrap, make_folds
from csi_medium_term_strategy import metrics
from random_forest_a_share_strategy import clean_for_json


@dataclass(frozen=True)
class RecencyScheme:
    scheme_id: int
    name: str
    change: str
    mode: str
    half_life: int | None = None
    lookback: int | None = None
    min_dates: int = 252


def recency_schemes() -> list[RecencyScheme]:
    return [
        RecencyScheme(1, "full_equal", "全历史等权训练基线。", "equal"),
        RecencyScheme(2, "exp_hl_126", "指数衰减，半衰期126个交易日，明显偏近期半年。", "exp", half_life=126),
        RecencyScheme(3, "exp_hl_252", "指数衰减，半衰期252个交易日，偏最近一年。", "exp", half_life=252),
        RecencyScheme(4, "exp_hl_504", "指数衰减，半衰期504个交易日，偏最近两年但保留长历史。", "exp", half_life=504),
        RecencyScheme(5, "rolling_504", "只使用最近504个交易日，约两年滚动训练。", "rolling", lookback=504),
        RecencyScheme(6, "rolling_756", "只使用最近756个交易日，约三年滚动训练。", "rolling", lookback=756),
    ]


def apply_recency_scheme(train: pd.DataFrame, scheme: RecencyScheme) -> tuple[pd.DataFrame, np.ndarray | None]:
    frame = train.copy()
    dates = pd.Index(sorted(frame["date"].unique()))
    if scheme.mode == "rolling":
        if scheme.lookback is None:
            raise ValueError(scheme.name)
        if len(dates) > scheme.lookback:
            frame = frame[frame["date"].isin(dates[-scheme.lookback :])].copy()
        if frame["date"].nunique() < scheme.min_dates:
            return frame.iloc[0:0].copy(), None
        return frame, None

    if scheme.mode == "equal":
        return frame, None

    if scheme.mode == "exp":
        if scheme.half_life is None:
            raise ValueError(scheme.name)
        date_rank = {date: i for i, date in enumerate(dates)}
        latest_rank = len(dates) - 1
        ages = frame["date"].map(lambda date: latest_rank - date_rank[date]).to_numpy(dtype=float)
        weights = np.power(0.5, ages / float(scheme.half_life))
        weights = weights / np.nanmean(weights)
        return frame, weights

    raise ValueError(scheme.mode)


def fit_weighted_model(train: pd.DataFrame, spec, cols: list[str], scheme: RecencyScheme) -> Pipeline | None:
    sample = train.dropna(subset=[spec.target]).copy()
    sample, weights = apply_recency_scheme(sample, scheme)
    if sample.empty or sample["date"].nunique() < scheme.min_dates:
        return None
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(sample):
            weights = weights[-len(sample) :]
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
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    pipe.fit(sample[cols], sample[spec.target], **fit_kwargs)
    return pipe


def select_candidate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: pd.DataFrame,
    schemes: list[RecencyScheme],
    exp_rules,
    score_mode: str,
    model_ids: set[int] | None,
    trade_rule_ids: set[int] | None,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    specs = model_specs(features)
    if model_ids:
        specs = [spec for spec in specs if spec.model_id in model_ids]
    trade_rules = diversified_rules(constrained_only=True)
    if trade_rule_ids:
        trade_rules = [rule for rule in trade_rules if rule.rule_id in trade_rule_ids]
    for scheme in schemes:
        for spec in specs:
            cols = columns_for_spec(features, spec)
            model = fit_weighted_model(train, spec, cols, scheme)
            if model is None:
                continue
            val_pred = predict_frame(model, val, cols)
            for trade_rule in trade_rules:
                for exp_rule in exp_rules:
                    val_bt, _ = dynamic_backtest(val_pred, trade_rule, exp_rule)
                    val_m = metrics(val_bt)
                    score = regime_score(val_m, score_mode)
                    row = {
                        "scheme_id": scheme.scheme_id,
                        "scheme_name": scheme.name,
                        "model_id": spec.model_id,
                        "feature_set": spec.feature_set,
                        "feature_count": len(cols),
                        "trade_rule_id": trade_rule.rule_id,
                        "exposure_name": exp_rule.name,
                        "target_vol": exp_rule.target_vol,
                        "score": score,
                        "scheme_change": scheme.change,
                        **{f"validation_{key}": value for key, value in val_m.items()},
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
    if best is None:
        raise RuntimeError("No candidate could be fit.")
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def evaluate_stress(equity: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bps in [14, 30, 50]:
        tmp = equity.copy()
        tmp["net_return"] = tmp["gross_return"] - tmp["turnover"] * bps / 10000
        rows.append({"case": f"cost_{bps}bps", "dropped_fold": None, **combined_metrics(tmp)})
    best_fold = int(folds.sort_values("test_sharpe", ascending=False).iloc[0]["fold_id"])
    worst_fold = int(folds.sort_values("test_sharpe").iloc[0]["fold_id"])
    last_fold = int(folds["fold_id"].max())
    for label, fold_id in [
        ("drop_best_fold", best_fold),
        ("drop_worst_fold", worst_fold),
        ("drop_last_fold", last_fold),
    ]:
        rows.append({"case": label, "dropped_fold": fold_id, **combined_metrics(equity[equity["fold_id"] != fold_id])})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    report_dir = REPORT_BASE / f"recency_n{args.max_symbols}_{args.report_name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    features = load_cached_features(args.max_symbols)
    features = attach_industry(features, Path(args.industry_map), args.industry_col)
    eligible = features.dropna(subset=["ret_60", "fwd_return_40"]).copy()
    dates = pd.Index(sorted(eligible["date"].unique()))
    folds = make_folds(dates, args.train_min, args.val_len, args.test_len, args.embargo)
    vol_budgets = [float(item) for item in args.vol_budgets.split(",")]
    exp_rules = exposure_candidates(vol_budgets)
    if args.exposure_names:
        wanted = {item.strip() for item in args.exposure_names.split(",") if item.strip()}
        exp_rules = [rule for rule in exp_rules if rule.name in wanted]
    schemes = recency_schemes()
    if args.schemes:
        wanted = {item.strip() for item in args.schemes.split(",") if item.strip()}
        schemes = [scheme for scheme in schemes if scheme.name in wanted]
    model_ids = {int(item) for item in args.model_ids.split(",") if item.strip()} if args.model_ids else None
    trade_rule_ids = {int(item) for item in args.trade_rule_ids.split(",") if item.strip()} if args.trade_rule_ids else None

    fold_rows = []
    selection_frames = []
    equity_frames = []
    holding_frames = []

    for fold_id, (train_dates, val_dates, test_dates) in enumerate(folds, start=1):
        print(
            f"[fold {fold_id}/{len(folds)}] train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"val {val_dates[0].date()}..{val_dates[-1].date()} test {test_dates[0].date()}..{test_dates[-1].date()}",
            flush=True,
        )
        train = features[features["date"].isin(train_dates)].copy()
        val = features[features["date"].isin(val_dates)].copy()
        test = features[features["date"].isin(test_dates)].copy()
        best, selection = select_candidate(train, val, features, schemes, exp_rules, args.score_mode, model_ids, trade_rule_ids)
        selection["fold_id"] = fold_id
        selection_frames.append(selection)

        train_full = features[features["date"].isin(dates[dates <= val_dates[-1]])].copy()
        model = fit_weighted_model(train_full, best["model"], best["cols"], best["scheme"])
        if model is None:
            raise RuntimeError(f"Could not refit selected scheme for fold {fold_id}")
        pred = predict_frame(model, test, best["cols"])
        bt, holdings = dynamic_backtest(pred, best["trade_rule"], best["exposure_rule"])
        bt["fold_id"] = fold_id
        holdings["fold_id"] = fold_id
        equity_frames.append(bt)
        holding_frames.append(holdings)
        test_m = metrics(bt)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "selected_scheme": best["scheme"].name,
                "selected_model_id": best["model"].model_id,
                "selected_feature_set": best["model"].feature_set,
                "selected_trade_rule_id": best["trade_rule"].rule_id,
                "selected_exposure_name": best["exposure_rule"].name,
                "selected_target_vol": best["exposure_rule"].target_vol,
                "validation_score": best["score"],
                "validation_sharpe": best["validation_sharpe"],
                "validation_annual_return": best["validation_annual_return"],
                "validation_max_drawdown": best["validation_max_drawdown"],
                "validation_avg_exposure": best["validation_avg_exposure"],
                **{f"test_{key}": value for key, value in test_m.items()},
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    selections = pd.concat(selection_frames, ignore_index=True)
    equity = pd.concat(equity_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)
    equity["combined_equity"] = (1 + equity["net_return"].fillna(0)).cumprod()
    combined = combined_metrics(equity)
    stress = evaluate_stress(equity, folds_df)
    returns = equity["net_return"].fillna(0)
    summary = {
        "objective": "Recency-weighted nested validation: validation folds choose full-history, exponential half-life, or rolling-window training.",
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
            "score_mode": args.score_mode,
            "vol_budgets": vol_budgets,
            "schemes": [scheme.name for scheme in schemes],
            "model_ids": sorted(model_ids) if model_ids else "all",
            "trade_rule_ids": sorted(trade_rule_ids) if trade_rule_ids else "all",
            "exposure_names": [rule.name for rule in exp_rules],
        },
        "combined": clean_for_json(combined),
        "folds": {
            "avg_test_sharpe": float(folds_df["test_sharpe"].mean()),
            "median_test_sharpe": float(folds_df["test_sharpe"].median()),
            "min_test_sharpe": float(folds_df["test_sharpe"].min()),
            "max_test_sharpe": float(folds_df["test_sharpe"].max()),
            "positive_folds": int((folds_df["test_sharpe"] > 0).sum()),
        },
        "bootstrap_sharpe": block_bootstrap(returns, args.bootstrap, args.block_len, args.seed),
        "selected_schemes": clean_for_json(folds_df["selected_scheme"].value_counts().to_dict()),
        "selected_exposure_rules": clean_for_json(folds_df["selected_exposure_name"].value_counts().to_dict()),
    }
    folds_df.to_csv(report_dir / "recency_folds.csv", index=False)
    selections.to_csv(report_dir / "recency_validation_candidates.csv", index=False)
    equity.to_csv(report_dir / "recency_equity.csv", index=False)
    holdings.to_csv(report_dir / "recency_holdings.csv", index=False)
    stress.to_csv(report_dir / "recency_stress_checks.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nRecency folds:")
    print(
        folds_df[
            [
                "fold_id",
                "selected_scheme",
                "selected_feature_set",
                "selected_trade_rule_id",
                "selected_exposure_name",
                "validation_sharpe",
                "test_sharpe",
                "test_annual_return",
                "test_max_drawdown",
                "test_avg_exposure",
            ]
        ].to_string(index=False)
    )
    print("\nSummary:")
    print(json.dumps(clean_for_json(summary), ensure_ascii=False, indent=2))
    print("Report:", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-symbols", type=int, default=200)
    parser.add_argument("--report-name", default="v1")
    parser.add_argument("--train-min", type=int, default=760)
    parser.add_argument("--val-len", type=int, default=194)
    parser.add_argument("--test-len", type=int, default=194)
    parser.add_argument("--embargo", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-mode", choices=["sharpe", "return_balance"], default="sharpe")
    parser.add_argument("--vol-budgets", default="0.18,0.22,0.26,0.30")
    parser.add_argument("--schemes", default="")
    parser.add_argument("--model-ids", default="")
    parser.add_argument("--trade-rule-ids", default="")
    parser.add_argument("--exposure-names", default="")
    parser.add_argument("--industry-map", default="data/processed/csi1000_industry_map_200.csv")
    parser.add_argument("--industry-col", default="industry_gate")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
