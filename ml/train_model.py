"""
Modelo de dos etapas para riesgo de incendio forestal:

  Etapa A (clasificación): ¿se producirá daño (área quemada > 0) dado
           el estado del combustible (índices FWI) y la meteorología?
  Etapa B (regresión):     si hay daño, ¿qué magnitud de área quemada
           (log1p ha) cabe esperar?

Esta separación en dos etapas es el enfoque habitual en la literatura de
riesgo de incendios (el propio dataset, Cortez & Morais 2007, señala que
~48% de los registros no tienen área > 0, lo que hace que una regresión
directa sobre todo el dataset esté dominada por ceros).

Uso:
    python -m ml.train_model
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from etl.transform import clean_fires_df
from ml.features import build_feature_matrix

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "forestfires.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

RANDOM_STATE = 42


def load_clean() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CSV)
    return clean_fires_df(raw)


def plot_feature_importance(model, feature_names, title: str, filename: str) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    importances.tail(12).plot.barh(ax=ax, color="#2980b9")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def train_classifier(df: pd.DataFrame, X: pd.DataFrame) -> dict:
    y = (df["burned_area_ha"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced")

    cv_f1 = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_f1_mean": round(float(cv_f1.mean()), 3),
        "cv_f1_std": round(float(cv_f1.std()), 3),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "test_precision": round(float(precision_score(y_test, y_pred)), 3),
        "test_recall": round(float(recall_score(y_test, y_pred)), 3),
        "test_f1": round(float(f1_score(y_test, y_pred)), 3),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_proba)), 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["sin daño", "con daño"], ax=ax, cmap="Blues"
    )
    ax.set_title("Clasificador: ¿habrá área quemada?")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "classifier_confusion_matrix.png", dpi=150)
    plt.close(fig)

    plot_feature_importance(
        clf, X.columns, "Importancia de variables - clasificador de ocurrencia", "classifier_feature_importance.png"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "fire_occurrence_classifier.joblib")

    return metrics


def train_regressor(df: pd.DataFrame, X: pd.DataFrame) -> dict:
    mask = df["burned_area_ha"] > 0
    X_damaged = X[mask.values].reset_index(drop=True)
    y_damaged = np.log1p(df.loc[mask, "burned_area_ha"]).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_damaged, y_damaged, test_size=0.2, random_state=RANDOM_STATE
    )

    reg = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=RANDOM_STATE)

    cv_mae = -cross_val_score(reg, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")

    reg.fit(X_train, y_train)
    y_pred_log = reg.predict(X_test)

    mae_log = mean_absolute_error(y_test, y_pred_log)
    rmse_log = mean_squared_error(y_test, y_pred_log) ** 0.5
    r2_log = r2_score(y_test, y_pred_log)

    y_test_ha = np.expm1(y_test)
    y_pred_ha = np.expm1(y_pred_log)
    mae_ha = mean_absolute_error(y_test_ha, y_pred_ha)

    metrics = {
        "cv_mae_log_mean": round(float(cv_mae.mean()), 3),
        "cv_mae_log_std": round(float(cv_mae.std()), 3),
        "test_mae_log_area": round(float(mae_log), 3),
        "test_rmse_log_area": round(float(rmse_log), 3),
        "test_r2_log_area": round(float(r2_log), 3),
        "test_mae_hectareas": round(float(mae_ha), 2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test_ha, y_pred_ha, alpha=0.6, color="#c0392b", edgecolors="k", linewidths=0.3)
    max_val = max(y_test_ha.max(), y_pred_ha.max())
    ax.plot([0, max_val], [0, max_val], "--", color="gray")
    ax.set_xlabel("área real quemada (ha)")
    ax.set_ylabel("área predicha (ha)")
    ax.set_title("Regresor: área quemada real vs. predicha\n(solo incendios con daño > 0)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regressor_real_vs_predicted.png", dpi=150)
    plt.close(fig)

    plot_feature_importance(
        reg, X.columns, "Importancia de variables - regresor de severidad", "regressor_feature_importance.png"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, MODELS_DIR / "fire_severity_regressor.joblib")

    return metrics


def run() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    X = build_feature_matrix(df)

    results = {
        "clasificador_ocurrencia": train_classifier(df, X),
        "regresor_severidad": train_regressor(df, X),
    }

    with open(METRICS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
