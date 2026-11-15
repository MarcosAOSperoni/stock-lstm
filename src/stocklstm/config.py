import os
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = Path(os.environ.get("STOCKLSTM_DATA_DIR", cfg["data_dir"]))
    cfg["reports_dir"] = Path(cfg.get("reports_dir", "reports"))
    if cfg.get("end_date") is None:
        cfg["end_date"] = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    return cfg


def raw_dir(cfg: dict) -> Path:
    path = cfg["data_dir"] / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(cfg: dict) -> Path:
    path = cfg["data_dir"] / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def holdout_start(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["end_date"]) - pd.DateOffset(months=cfg["holdout_months"])
