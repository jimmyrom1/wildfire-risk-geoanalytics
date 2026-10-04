from pathlib import Path

import pandas as pd
import pytest

from etl.transform import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, clean_fires_df, grid_to_latlon

RAW_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "forestfires.csv"


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def test_clean_fires_df_drops_duplicates_and_nulls(raw_df):
    clean = clean_fires_df(raw_df)
    assert len(clean) <= len(raw_df)
    assert clean.isna().sum().sum() == 0


def test_clean_fires_df_expected_columns(raw_df):
    clean = clean_fires_df(raw_df)
    expected = {
        "x_grid", "y_grid", "month", "day_of_week", "ffmc", "dmc", "dc", "isi",
        "temp_c", "rh_pct", "wind_kmh", "rain_mm", "burned_area_ha", "lat", "lon",
    }
    assert expected.issubset(set(clean.columns))


def test_clean_fires_df_coordinates_within_park_bbox(raw_df):
    clean = clean_fires_df(raw_df)
    assert clean["lat"].between(LAT_MIN, LAT_MAX).all()
    assert clean["lon"].between(LON_MIN, LON_MAX).all()


def test_grid_to_latlon_extremes_map_to_bbox_corners():
    lat, lon = grid_to_latlon(pd.Series([1]), pd.Series([1]))
    assert lat.iloc[0] == LAT_MIN
    assert lon.iloc[0] == LON_MIN

    lat, lon = grid_to_latlon(pd.Series([9]), pd.Series([9]))
    assert lat.iloc[0] == LAT_MAX
    assert lon.iloc[0] == LON_MAX


def test_clean_fires_df_raises_on_missing_columns():
    bad_df = pd.DataFrame({"X": [1], "Y": [1]})
    with pytest.raises(ValueError):
        clean_fires_df(bad_df)
