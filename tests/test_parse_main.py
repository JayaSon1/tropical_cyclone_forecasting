from pathlib import Path

import pandas as pd

from config import settings

# parse_hurdat2's entry point reads its input and output paths from config, not hard-coded
# relative paths (repo hygiene, after AUDIT.md §8 session 6).
ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "irene_2011.txt"


def test_config_paths_are_the_repo_defaults():
    assert settings.RAW_HURDAT2_PATH == ROOT / "data" / "raw" / "hurdat2-1851-2025-02272026.txt"
    assert settings.RAW_PARQUET_PATH == ROOT / "data" / "processed" / "hurdat2_raw.parquet"
    assert settings.PROCESSED_DATA_PATH == ROOT / "data" / "processed" / "hurdat2_processed_observations.parquet"


def test_main_writes_to_config_paths(tmp_path, monkeypatch):
    from data import parse_hurdat2
    raw_out = tmp_path / "out" / "raw.parquet"
    processed_out = tmp_path / "out" / "processed.parquet"
    monkeypatch.setattr(settings, "RAW_HURDAT2_PATH", FIXTURE)
    monkeypatch.setattr(settings, "RAW_PARQUET_PATH", raw_out)
    monkeypatch.setattr(settings, "PROCESSED_DATA_PATH", processed_out)
    monkeypatch.chdir(tmp_path)   # relative paths would land here

    parse_hurdat2.main()

    assert (pd.read_parquet(raw_out)["storm_id"] == "AL092011").all()
    processed = pd.read_parquet(processed_out)
    assert set(settings.FEATURE_COLS + ["RI"]) <= set(processed.columns)
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert written == ["out/processed.parquet", "out/raw.parquet"]
