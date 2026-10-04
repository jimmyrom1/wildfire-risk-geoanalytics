"""
Análisis exploratorio de datos (EDA) sobre el dataset de incendios.

No depende de PostgreSQL: lee directamente el CSV limpio, igual que
haría un análisis exploratorio previo a decidir el diseño de la base de
datos y del modelo. Genera figuras reales en reports/figures/ y un
resumen de hallazgos en reports/metrics/eda_summary.json.

Uso:
    python -m analysis.eda
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from etl.transform import clean_fires_df

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "forestfires.csv"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

sns.set_theme(style="whitegrid")

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def load_clean() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CSV)
    return clean_fires_df(raw)


def plot_area_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(df["burned_area_ha"], bins=40, color="#c0392b")
    axes[0].set_title("Área quemada (ha) - escala original")
    axes[0].set_xlabel("hectáreas")

    log_area = np.log1p(df["burned_area_ha"])
    axes[1].hist(log_area, bins=40, color="#2c3e50")
    axes[1].set_title("Área quemada - log(1 + ha)")
    axes[1].set_xlabel("log(1 + hectáreas)")

    fig.suptitle("Distribución del área quemada por incendio")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "area_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = ["ffmc", "dmc", "dc", "isi", "temp_c", "rh_pct", "wind_kmh", "rain_mm", "burned_area_ha"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlación: índices FWI, clima y área quemada")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_seasonality(df: pd.DataFrame) -> None:
    by_month = (
        df.groupby("month")
        .agg(n_incendios=("burned_area_ha", "size"), area_total=("burned_area_ha", "sum"))
        .reindex(MONTH_ORDER)
        .dropna(how="all")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(by_month.index, by_month["n_incendios"], color="#d35400")
    axes[0].set_title("Nº de incendios por mes")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(by_month.index, by_month["area_total"], color="#8e44ad")
    axes[1].set_title("Área total quemada por mes (ha)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "seasonality.png", dpi=150)
    plt.close(fig)


def plot_day_of_week(df: pd.DataFrame) -> None:
    by_day = df.groupby("day_of_week").size().reindex(DAY_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_day.index, by_day.values, color="#16a085")
    ax.set_title("Nº de incendios por día de la semana")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "by_day_of_week.png", dpi=150)
    plt.close(fig)


def plot_geo_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        df["lon"], df["lat"],
        s=20 + df["burned_area_ha"] * 2,
        c=df["burned_area_ha"],
        cmap="autumn_r",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_title("Localización aproximada de incendios\n(tamaño/color = área quemada)")
    ax.set_xlabel("longitud (aprox.)")
    ax.set_ylabel("latitud (aprox.)")
    fig.colorbar(sc, ax=ax, label="área quemada (ha)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geo_scatter.png", dpi=150)
    plt.close(fig)


def compute_summary(df: pd.DataFrame) -> dict:
    no_damage = (df["burned_area_ha"] == 0).mean()
    by_month_area = df.groupby("month")["burned_area_ha"].sum().sort_values(ascending=False)
    corr_temp_area = df["temp_c"].corr(df["burned_area_ha"])
    corr_isi_area = df["isi"].corr(df["burned_area_ha"])

    return {
        "n_registros": int(len(df)),
        "pct_incendios_sin_area_registrada": round(float(no_damage) * 100, 1),
        "mes_mas_destructivo": by_month_area.index[0],
        "area_total_mes_mas_destructivo_ha": round(float(by_month_area.iloc[0]), 2),
        "correlacion_temp_area": round(float(corr_temp_area), 3),
        "correlacion_isi_area": round(float(corr_isi_area), 3),
        "area_media_ha": round(float(df["burned_area_ha"].mean()), 2),
        "area_mediana_ha": round(float(df["burned_area_ha"].median()), 2),
        "area_maxima_ha": round(float(df["burned_area_ha"].max()), 2),
    }


def run() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean()

    plot_area_distribution(df)
    plot_correlation_heatmap(df)
    plot_seasonality(df)
    plot_day_of_week(df)
    plot_geo_scatter(df)

    summary = compute_summary(df)
    with open(METRICS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
