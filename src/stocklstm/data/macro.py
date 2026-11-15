from io import StringIO

import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def download_fred(series: dict[str, str], start: str) -> pd.DataFrame:
    """Wide frame indexed by observation date, one column per configured FRED series."""
    cols = {}
    for name, sid in series.items():
        resp = requests.get(FRED_URL.format(sid=sid), timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), na_values=".")
        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"])
        cols[name] = df.set_index("date")[name]
    out = pd.DataFrame(cols).sort_index()
    return out[out.index >= pd.Timestamp(start)]
