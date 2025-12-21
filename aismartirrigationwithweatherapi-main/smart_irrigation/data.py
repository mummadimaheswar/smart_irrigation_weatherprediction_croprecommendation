"""Load and clean sensor CSV data."""
import re
import pandas as pd


def _norm(name: str) -> str:
    """Normalise column name to lowercase with underscores."""
    s = name.strip().lower()
    s = re.sub(r"[%()/ -]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _find_soil_col(df: pd.DataFrame) -> str:
    """Find the best soil moisture column."""
    nmap = {_norm(c): c for c in df.columns}
    for key in ("soil_moisture", "volume_soilmoisture_percentage_at_15cm",
                "aggregate_soilmoisture_percentage_at_15cm"):
        if key in nmap:
            return nmap[key]
    raise ValueError("No soil moisture column found.")


def load_and_clean(path: str, state: str | None = None, district: str | None = None) -> pd.DataFrame:
    """Load CSV, normalise columns, filter, and return a clean DataFrame."""
    df = pd.read_csv(path)
    if df.empty:
        return df

    # Rename common columns
    renames = {}
    nmap = {_norm(c): c for c in df.columns}
    for alias, target in [("date", "date"), ("datetime", "date"), ("state_name", "state"),
                           ("districtname", "district")]:
        if alias in nmap and nmap[alias] != target:
            renames[nmap[alias]] = target
    df = df.rename(columns=renames)

    if "date" not in df.columns:
        raise ValueError("Missing date column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    soil_col = _find_soil_col(df)
    df["soil_moisture"] = pd.to_numeric(df[soil_col], errors="coerce")
    df = df.dropna(subset=["soil_moisture"])
    if df["soil_moisture"].max() > 1.5:
        df["soil_moisture"] /= 100.0

    if state and "state" in df.columns:
        df = df[df["state"].str.lower() == state.lower()]
    if district and "district" in df.columns:
        df = df[df["district"].str.lower() == district.lower()]

    return df.sort_values("date").reset_index(drop=True)
