"""
Cluster cities into archetypes from daily features.
Outputs:
  data/processed/clustering/city_features.parquet
  data/processed/clustering/city_clusters.parquet
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.data.utils import load_config
from src.utils.logger import setup_logger

def build_city_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("city").agg(
    health_mean=("health_score","mean"),
    health_std =("health_score","std"),
    air_mean=("air_score","mean"),
    temp_mean=("temp_score","mean"),
    precip_mean=("precip_score","mean"),
    wind_mean=("wind_score","mean"),
    n_days     =("date","nunique")
    ).reset_index()

    return agg

def main():
    logger = setup_logger("city_clustering")
    cfg = load_config()
    processed = Path(cfg["paths"]["processed"])
    outdir = processed / "clustering"
    outdir.mkdir(parents=True, exist_ok=True)

    health = pd.read_parquet(processed / "citypulse_health_daily.parquet")
    health["date"] = pd.to_datetime(health["date"])

    feats = build_city_features(health)
    feats.to_parquet(outdir / "city_features.parquet", index=False)

    # Prepare for clustering
    Xcols = ["health_mean","health_std","air_mean","temp_mean","precip_mean","wind_mean"]
    X = feats[Xcols].fillna(feats[Xcols].median())
    Xs = StandardScaler().fit_transform(X)

    # Choose K by eyeballing (tiny N), K=3 works well for 5 cities
    km = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = km.fit_predict(Xs)
    clusters = feats[["city"]].copy()
    clusters["cluster"] = labels
    clusters.to_parquet(outdir / "city_clusters.parquet", index=False)

    logger.info(f"Clusters:\n{clusters}")

    # Quick 2D viz (PCA-like via first 2 standardized features)
    plt.figure(figsize=(6,5))
    for k in sorted(np.unique(labels)):
        idx = labels==k
        plt.scatter(Xs[idx,0], Xs[idx,1], label=f"Cluster {k}", s=90)
    for i, name in enumerate(feats["city"]):
        plt.text(Xs[i,0]+0.02, Xs[i,1]+0.02, name, fontsize=9)
    plt.title("City Clusters (quick 2D projection)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "clusters_quickplot.png", dpi=180)
    logger.info(f"Saved: {outdir / 'clusters_quickplot.png'}")

if __name__ == "__main__":
    main()
