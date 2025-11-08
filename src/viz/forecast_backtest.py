"""
forecast_backtest.py
--------------------
show backtest predictions vs truth (one city)
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

preds_path = Path("data/processed/forecast/backtest_predictions.parquet")

df = pd.read_parquet(preds_path)
df["date"] = pd.to_datetime(df["date"])

city = "Casablanca"  # you can change or loop
subset = df[df.city == city]

plt.figure(figsize=(14,6))
sns.lineplot(data=subset, x="date", y="y_true", color="black", label="True")
sns.lineplot(data=subset, x="date", y="y_pred", hue="model", alpha=0.5)
plt.title(f"Rolling Backtest — {city} — True vs Predictions")
plt.ylabel("Health Score")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
