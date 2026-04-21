"""Assertions against the built gazetteer artifacts.

Run ``python etl/build_gazetteer.py --xml data/full_database.xml`` first;
these tests load the pickles produced at the repo root.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from rapidfuzz import fuzz, process

REPO_ROOT = Path(__file__).resolve().parent.parent
EXACT_PATH = REPO_ROOT / "gazetteer.pkl"
FUZZY_PATH = REPO_ROOT / "gazetteer_fuzzy.pkl"


@pytest.fixture(scope="module")
def exact() -> dict[str, str]:
    if not EXACT_PATH.exists():
        pytest.skip(f"{EXACT_PATH} not found — run build_gazetteer.py first")
    with EXACT_PATH.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def fuzzy() -> list[tuple[str, str]]:
    if not FUZZY_PATH.exists():
        pytest.skip(f"{FUZZY_PATH} not found — run build_gazetteer.py first")
    with FUZZY_PATH.open("rb") as f:
        return pickle.load(f)


def test_exact_warfarin(exact: dict[str, str]) -> None:
    assert exact.get("coumadin") == "DB00682"
    assert exact.get("warfarin") == "DB00682"


def test_exact_metformin(exact: dict[str, str]) -> None:
    assert exact.get("glucophage") == "DB00331"
    assert exact.get("metformin") == "DB00331"


def test_exact_atorvastatin(exact: dict[str, str]) -> None:
    assert exact.get("lipitor") == "DB01076"
    assert exact.get("atorvastatin") == "DB01076"


def test_fuzzy_warfarin_typo(fuzzy: list[tuple[str, str]]) -> None:
    choices = [name for name, _ in fuzzy]
    id_by_name = {name: dbid for name, dbid in fuzzy}
    hit = process.extractOne(
        "warfarn", choices, scorer=fuzz.token_set_ratio, score_cutoff=85
    )
    assert hit is not None, "expected 'warfarn' to fuzzy-match an indexed name"
    matched_name = hit[0]
    assert id_by_name[matched_name] == "DB00682"
