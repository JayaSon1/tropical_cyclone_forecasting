from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.features.build_features import extract_features, point_distance_km
from src.features.labels import ri_labels

# Label scope and time base (AUDIT.md §6 decisions 1, 4, 5; §8 session 4):
# only synoptic rows (00/06/12/18Z) are labelled; t+24 and the 6 h / 12 h lags are
# exact-timestamp matches within the storm; rows whose status at t or t+24 is not
# tropical are excluded; a row is excluded if any raw L record falls in [t, t+24 h].
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hurdat2-1851-2025-02272026.txt"
T0 = pd.Timestamp("2020-08-01 00:00")


def make(points, storm_id="AL012020"):
    # points: (hours after T0, vmax[, status[, record_id]])
    rows = []
    for p in points:
        h, vmax, status, record_id = (list(p) + ["HU", None][len(p) - 2:])[:4]
        rows.append({
            "storm_id": storm_id, "name": "TEST", "datetime": T0 + pd.Timedelta(hours=h),
            "record_id": record_id, "status": status,
            "latitude": 20.0 + 0.1 * h, "longitude": -60.0 - 0.2 * h,
            "vmax": vmax, "mslp": 1000 - (vmax or 0) // 2,
        })
    return pd.DataFrame(rows)


def row_at(df, h, storm_id="AL012020"):
    out = df[(df["storm_id"] == storm_id) & (df["datetime"] == T0 + pd.Timedelta(hours=h))]
    assert len(out) == 1, f"expected one row at t+{h} h, got {len(out)}"
    return out.iloc[0]


REGULAR = [0, 6, 12, 18, 24, 30]


# ---------- labels ----------

@pytest.mark.parametrize("delta, expected", [(29, 0), (30, 1), (31, 1)])
def test_ri_threshold_boundary(delta, expected):
    df = ri_labels(make([(h, 40 + (delta if h == 24 else 0)) for h in REGULAR]))
    assert row_at(df, 0)["RI"] == expected


def test_off_synoptic_record_dropped_and_t24_by_time():
    # An intensity-peak record at 03Z must not shift t+24 onto the 18Z row
    df = ri_labels(make([(0, 40), (3, 45, "HU", "I"), (6, 50), (12, 55), (18, 60), (24, 75), (30, 80)]))
    assert not (df["datetime"] == T0 + pd.Timedelta(hours=3)).any()
    t0 = row_at(df, 0)
    assert t0["vmax_24h"] == 75
    assert t0["RI"] == 1


def test_missing_synoptic_step_gives_na():
    # No record at t+24: the label is NA even though a record exists 4 rows later
    df = ri_labels(make([(0, 40), (6, 45), (12, 50), (18, 55), (30, 90), (36, 90)]))
    assert pd.isna(row_at(df, 0)["RI"])
    assert row_at(df, 6)["vmax_24h"] == 90 and row_at(df, 6)["RI"] == 1


@pytest.mark.parametrize("status_24, labelled", [
    ("EX", False), ("LO", False), ("WV", False), ("DB", False),
    ("TD", True), ("TS", True), ("HU", True), ("SD", True), ("SS", True),
])
def test_status_at_t24_must_be_tropical(status_24, labelled):
    df = ri_labels(make([(0, 40), (6, 45), (12, 50), (18, 55), (24, 60, status_24), (30, 60)]))
    assert pd.notna(row_at(df, 0)["RI"]) == labelled


def test_tropical_statuses_single_constant():
    assert settings.TROPICAL_STATUSES == ["TD", "TS", "HU", "SD", "SS"]


@pytest.mark.parametrize("points, excluded", [
    # off-synoptic L at t+3 h
    ([(0, 40), (3, 42, "HU", "L"), (6, 45), (12, 50), (18, 55), (24, 60), (30, 60)], True),
    # L at t+21 h behind two off-synoptic records: outside today's 4-row window, inside 24 h
    ([(0, 40), (3, 42, "HU", "I"), (6, 45), (9, 47, "HU", "I"), (12, 50), (18, 55),
      (21, 57, "HU", "L"), (24, 60), (30, 60)], True),
    # synoptic L at exactly t+24 h (window is closed)
    ([(0, 40), (6, 45), (12, 50), (18, 55), (24, 60, "HU", "L"), (30, 60)], True),
    # synoptic L at t
    ([(0, 40, "HU", "L"), (6, 45), (12, 50), (18, 55), (24, 60), (30, 60)], True),
    # L at t+27 h, after a gap: inside today's 4-row window, outside 24 h
    ([(0, 40), (6, 45), (12, 50), (24, 60), (27, 62, "HU", "L"), (30, 60)], False),
])
def test_landfall_window_by_time(points, excluded):
    df = ri_labels(make(points))
    assert pd.isna(row_at(df, 0)["RI"]) == excluded


def test_storms_never_mix():
    a = make([(0, 40), (6, 40), (12, 40), (18, 40)], storm_id="AL012020")
    b = make([(24, 120), (30, 120), (36, 120), (42, 120), (48, 120)], storm_id="AL022020")
    df = ri_labels(pd.concat([a, b], ignore_index=True))
    assert df.loc[df["storm_id"] == "AL012020", "RI"].isna().all()


def test_missing_vmax_gives_na():
    df = ri_labels(make([(0, None), (6, 40), (12, 45), (18, 50), (24, 60), (30, None), (36, 70)]))
    assert pd.isna(row_at(df, 0)["RI"])   # vmax missing at t
    assert pd.isna(row_at(df, 6)["RI"])   # vmax missing at t+24


def test_duplicate_timestamp_raises():
    df = make([(0, 40), (6, 45), (6, 45), (12, 50), (18, 55), (24, 60)])
    with pytest.raises(ValueError, match="AL012020"):
        ri_labels(df)


def test_value_at_exact_match():
    from src.features.time_utils import value_at
    df = make([(0, 40), (6, 45), (18, 55), (24, 60)])
    np.testing.assert_array_equal(value_at(df, 6, "vmax"), [45, np.nan, 60, np.nan])
    np.testing.assert_array_equal(value_at(df, -6, "vmax"), [np.nan, 40, np.nan, 55])


# ---------- features ----------

def features(points, **kw):
    return extract_features(ri_labels(make(points, **kw)))


def test_lags_use_exact_offsets():
    df = features([(0, 40), (6, 45), (9, 70, "HU", "I"), (12, 50), (18, 55)])
    t12 = row_at(df, 12)
    assert t12["delta_vmax_6h"] == 50 - 45
    assert t12["delta_vmax_12h"] == 50 - 40
    gap = features([(0, 40), (6, 45), (18, 55), (24, 60)])
    assert pd.isna(row_at(gap, 18)["delta_vmax_6h"])
    assert row_at(gap, 18)["delta_vmax_12h"] == 55 - 45


def test_translation_speed_uses_t_minus_6h():
    df = features([(0, 40), (6, 45), (9, 70, "HU", "I"), (12, 50), (18, 55)])
    expected = point_distance_km(20.6, -61.2, 21.2, -62.4) / 6.0
    assert row_at(df, 12)["translation_speed"] == pytest.approx(expected, rel=1e-12)
    gap = features([(0, 40), (6, 45), (18, 55), (24, 60)])
    assert pd.isna(row_at(gap, 18)["translation_speed"])


def test_features_at_t_ignore_the_future():
    points = [(h, 40 + h) for h in range(0, 49, 6)]
    base = features(points)
    changed = make(points)
    later = changed["datetime"] > T0 + pd.Timedelta(hours=18)
    changed.loc[later, ["vmax", "mslp", "latitude", "longitude"]] = [150, 900, 45.0, -20.0]
    after = extract_features(ri_labels(changed))
    cols = settings.FEATURE_COLS
    mask = base["datetime"] <= T0 + pd.Timedelta(hours=18)
    pd.testing.assert_frame_equal(base.loc[mask, cols].reset_index(drop=True),
                                  after.loc[mask, cols].reset_index(drop=True))


def test_no_future_columns_in_features():
    future = {"vmax_24h", "delta_vmax_24h", "status_24h", "landfall_next_24h", "RI"}
    assert not future & set(settings.FEATURE_COLS)


# ---------- full data (Check A/B simulation, AUDIT.md §7) ----------

def test_full_data_matches_check_ab_simulation():
    if not RAW.exists():
        pytest.skip("raw HURDAT2 file missing")
    from data.parse_hurdat2 import parse_hurdat2, select_modelling_rows
    from src.models.xgboost_model import split_data

    rows = select_modelling_rows(extract_features(ri_labels(parse_hurdat2(str(RAW)))))
    assert (len(rows), int(rows["RI"].sum())) == (12941, 858)
    train, val, test = split_data(rows)
    counts = [(len(s), int(s["RI"].sum())) for s in (train, val, test)]
    # Train is 1980-2015 (§6 decision 12): Check B 1980+ (10,288 / 670) minus 2016-19 and 2020+
    assert counts == [(7684, 468), (1088, 88), (1516, 114)]
