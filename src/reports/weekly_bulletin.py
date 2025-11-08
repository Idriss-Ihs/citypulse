"""
weekly_bulletin.py
------------------
Generate weekly CityPulse bulletin PDF.
"""

from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import argparse

DATA_ACTUAL = Path("data\processed\citypulse_events_full.parquet")
DATA_FORECAST = Path("data/processed/forecast/backtest_predictions.parquet")
OUT_DIR = Path("data/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists


def generate_bulletin(week_start=None):
    df = pd.read_parquet(DATA_ACTUAL)
    df_fc = pd.read_parquet(DATA_FORECAST)

    df["date"] = pd.to_datetime(df["date"])
    df_fc["date"] = pd.to_datetime(df_fc["date"])

    if week_start is None:
        # last full week in dataset
        last_date = df["date"].max()
        week_start = last_date - pd.Timedelta(days=6)
        week_end = last_date
    else:
        week_start = pd.to_datetime(week_start)
        week_end = week_start + pd.Timedelta(days=6)

    df_week = df[df["date"].between(week_start, week_end)]
    weekly = df_week.groupby("city")["health_score"].mean().reset_index()
    weekly.columns = ["City", "Weekly_Health"]

    top_city = weekly.sort_values("Weekly_Health", ascending=False).iloc[0]
    bottom_city = weekly.sort_values("Weekly_Health", ascending=True).iloc[0]

    # forecast next 7 days
    df_fc_week = df_fc[df_fc["date"] > week_end]
    next7 = df_fc_week.groupby("city")["y_pred"].mean().reset_index()
    next7.columns = ["City", "Forecast_Weekly"]

    # PDF path
    OUTPUT_PDF = OUT_DIR / f"citypulse_weekly_{week_start.date()}_{week_end.date()}.pdf"

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>CityPulse — Weekly Urban Health Bulletin</b>", styles["Title"]))
    story.append(Spacer(1, 14))

    summary_txt = f"""
    Period: {week_start.date()} → {week_end.date()}<br/><br/>
    Highest weekly health: <b>{top_city['City']}</b> ({top_city['Weekly_Health']:.1f})<br/>
    Lowest weekly health: <b>{bottom_city['City']}</b> ({bottom_city['Weekly_Health']:.1f})<br/><br/>
    Next 7 days are projected from a rolling multi-model forecast ensemble.
    """

    story.append(Paragraph(summary_txt, styles["BodyText"]))
    story.append(Spacer(1, 22))

    # table: observed
    table1 = Table([["City", "Weekly Health"]] + weekly.values.tolist())
    table1.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
    ]))
    story.append(Paragraph("<b>Last 7 days</b>", styles["Heading2"]))
    story.append(table1)
    story.append(Spacer(1, 18))

    # table: forecast
    table2 = Table([["City", "Forecast Next 7 days"]] + next7.values.tolist())
    table2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.25,colors.grey),
    ]))
    story.append(Paragraph("<b>Next 7 days forecast</b>", styles["Heading2"]))
    story.append(table2)

    doc = SimpleDocTemplate(str(OUTPUT_PDF))
    doc.build(story)

    print(f"✅ weekly bulletin created: {OUTPUT_PDF}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=str, default=None, help="week start date YYYY-MM-DD")
    args = parser.parse_args()
    generate_bulletin(args.week)
