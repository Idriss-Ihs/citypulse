import pandas as pd
import yaml
from pathlib import Path


def load_config(path="src/config/settings.yaml"):
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_data(df: pd.DataFrame, name: str, folder: str | Path):
    """
    Save a dataframe to CSV in a specific folder.
    Example: save_data(df, "weather", "data/raw")
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{name}.csv"
    df.to_csv(out_path, index=False)
    return out_path
