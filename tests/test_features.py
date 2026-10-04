from pathlib import Path

import pandas as pd

from etl.transform import clean_fires_df
from ml.features import build_feature_matrix

RAW_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "forestfires.csv"


def test_build_feature_matrix_shape_and_no_nulls():
    raw = pd.read_csv(RAW_CSV)
    clean = clean_fires_df(raw)
    X = build_feature_matrix(clean)

    assert len(X) == len(clean)
    assert X.isna().sum().sum() == 0
    assert "ffmc" in X.columns
    assert any(col.startswith("month_") for col in X.columns)
    assert any(col.startswith("day_of_week_") for col in X.columns)
