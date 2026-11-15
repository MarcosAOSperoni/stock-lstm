import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import holdout_start, processed_dir, raw_dir  # noqa: E402
from .dataset import coverage_summary, dataset_path, feature_columns  # noqa: E402
from .evaluate import daily_ic, hit_rate, ic_summary, long_short_backtest  # noqa: E402
from .models.baselines import (  # noqa: E402
    FACTOR_BASELINES,
    factor_predictions,
    fit_lightgbm,
    predict_lightgbm,
)
from .splits import walk_forward_folds  # noqa: E402

log = logging.getLogger(__name__)

MODELS = [*FACTOR_BASELINES, "lightgbm"]


def run_baselines(cfg: dict, horizons: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path(cfg))
    features = feature_columns(df)
    hold = holdout_start(cfg)
    dates = pd.DatetimeIndex(df["date"].unique())
    wf, lgb_cfg = cfg["walk_forward"], cfg["lightgbm"]
    horizons = horizons or list(cfg["horizons"])

    preds, importances, metrics, ics = [], [], [], {}
    for name in horizons:
        h = cfg["horizons"][name]
        y, fwd = f"y_{name}", f"fwd_{name}"
        folds = walk_forward_folds(
            dates,
            cfg["start_date"],
            hold,
            wf["train_years"],
            wf["test_months"],
            purge=h + cfg["execution_lag"],
        )
        log.info("horizon %s: %d folds", name, len(folds))
        for i, fold in enumerate(folds):
            train = df[(df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)]
            train = train.dropna(subset=[y])
            test = df[(df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)]
            model = fit_lightgbm(train, features, y, lgb_cfg["params"], lgb_cfg["num_boost_round"])
            p = test[["date", "ticker", y, fwd, f"fwd_bench_{name}"]].rename(
                columns={y: "y", fwd: "fwd", f"fwd_bench_{name}": "fwd_bench"}
            )
            p = p.join(factor_predictions(test))
            p["lightgbm"] = predict_lightgbm(model, test, features)
            preds.append(p.assign(horizon=name, fold=i))
            importances.append(
                pd.DataFrame(
                    {
                        "feature": features,
                        "gain": model.feature_importance("gain"),
                        "horizon": name,
                        "fold": i,
                    }
                )
            )
            log.info(
                "  fold %d train %s..%s test %s..%s (%d train rows)",
                i,
                fold.train_start.date(),
                fold.train_end.date(),
                fold.test_start.date(),
                fold.test_end.date(),
                len(train),
            )
        oos = pd.concat([p for p in preds if p["horizon"].iat[0] == name])
        for model_name in MODELS:
            ic = daily_ic(oos, model_name, "y")
            ics[(name, model_name)] = ic
            row = {"horizon": name, "model": model_name, **ic_summary(ic, h)}
            row["hit_rate"] = hit_rate(oos, model_name, "y")
            row.update(
                long_short_backtest(
                    oos,
                    model_name,
                    "fwd",
                    h,
                    cfg["backtest"]["quantile"],
                    cfg["backtest"]["cost_bps"],
                    bench="fwd_bench",
                )
            )
            metrics.append(row)

    out = pd.DataFrame(metrics)
    pd.concat(preds).to_parquet(processed_dir(cfg) / "oos_predictions.parquet", index=False)
    write_report(cfg, out, ics, pd.concat(importances), Path(cfg["reports_dir"]) / "baselines")
    return out


def write_report(
    cfg: dict, metrics: pd.DataFrame, ics: dict, importances: pd.DataFrame, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    imp = (
        importances.groupby(["horizon", "feature"])["gain"]
        .mean()
        .reset_index()
        .sort_values(["horizon", "gain"], ascending=[True, False])
    )
    imp.to_csv(out_dir / "feature_importance.csv", index=False)

    horizons = list(dict.fromkeys(metrics["horizon"]))
    fig, axes = plt.subplots(len(horizons), 1, figsize=(10, 3 * len(horizons)), squeeze=False)
    for ax, name in zip(axes[:, 0], horizons, strict=True):
        for model in MODELS:
            ics[(name, model)].cumsum().plot(ax=ax, label=model)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"Cumulative daily IC, horizon {name}")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_ic.png", dpi=110)
    plt.close(fig)

    cov_path = raw_dir(cfg) / "coverage.csv"
    coverage = coverage_summary(pd.read_csv(cov_path)) if cov_path.exists() else None
    cols = {
        "horizon": "horizon",
        "model": "model",
        "mean_ic": "mean IC",
        "ic_t_stat": "IC t-stat",
        "hit_rate": "hit rate",
        "long_short_gross_ann_return": "L/S gross ann. return",
        "long_short_ann_return": "L/S net ann. return",
        "long_short_sharpe": "L/S Sharpe",
        "long_short_max_drawdown": "L/S max DD",
        "long_only_ann_return": "top-decile excess ann.",
        "avg_turnover": "turnover",
        "n_periods": "periods",
    }
    table = metrics[list(cols)].rename(columns=cols)
    lines = [
        "# Baseline results (walk-forward, out-of-sample, pre-holdout)",
        "",
        f"- Study period: {cfg['start_date']} to {cfg['end_date']}; "
        f"holdout from {holdout_start(cfg).date()} (not used).",
        f"- Labels: excess log return vs {cfg['market']['benchmark']}, entry "
        f"{cfg['execution_lag']} day(s) after signal, ranked among index members.",
        f"- Backtest: top vs bottom {cfg['backtest']['quantile']:.0%}, equal weight, "
        f"{cfg['backtest']['cost_bps']} bps per unit turnover, rebalanced every horizon.",
    ]
    if coverage:
        lines.append(f"- Price coverage: {json.dumps(coverage)}")
    lines += ["", table.to_markdown(index=False, floatfmt=".4f"), ""]
    (out_dir / "metrics.md").write_text("\n".join(lines))
