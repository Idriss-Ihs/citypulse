"""
forecast_leaderboard.py
-----------------------
Plot leaderboard (RMSE) per model per city.
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

metrics_path = Path("data/processed/forecast/metrics_cv.csv")

df = pd.read_csv(metrics_path)

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="model", y="RMSE", hue="city")
plt.title("Rolling-Origin CV — Models RMSE per City (lower is better)")
plt.ylabel("RMSE")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
