import argparse
import logging

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stocklstm")
    parser.add_argument("--config", default=None, help="path to YAML config")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download membership, prices, market and FRED data")
    build = sub.add_parser("build-dataset", help="build features and labels")
    build.add_argument("--max-tickers", type=int, default=None, help="random subset for quick runs")
    base = sub.add_parser("baselines", help="walk-forward baselines and report")
    base.add_argument("--horizons", default=None, help="comma-separated, e.g. 1d,1m")
    run = sub.add_parser("run", help="download, build-dataset and baselines")
    run.add_argument("--max-tickers", type=int, default=None)
    run.add_argument("--horizons", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = load_config(args.config)

    from .dataset import build_dataset, dataset_path, download_all
    from .pipeline import run_baselines

    horizons = args.horizons.split(",") if getattr(args, "horizons", None) else None
    if args.cmd in ("download", "run"):
        download_all(cfg)
    if args.cmd in ("build-dataset", "run"):
        df = build_dataset(cfg, max_tickers=args.max_tickers)
        df.to_parquet(dataset_path(cfg), index=False)
        logging.info("dataset: %d rows x %d columns", *df.shape)
    if args.cmd in ("baselines", "run"):
        metrics = run_baselines(cfg, horizons)
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
