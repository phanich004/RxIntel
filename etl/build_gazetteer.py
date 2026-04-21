"""
build_gazetteer.py
==================
Build a rapidfuzz-friendly drug-name gazetteer from the DrugBank XML.

Output artifacts (repo root by default):
  - gazetteer.pkl        dict[str, str]        normalized string -> DrugBank ID
  - gazetteer_fuzzy.pkl  list[(str, str)]      (normalized, drug_id) pairs
                                                 for rapidfuzz.process.extractOne

Only approved+investigational drugs are indexed (matches parse_drugbank.py
and build_vector_store.py). For each drug we collect:
  * primary <name>
  * every <synonyms>/<synonym>
  * every unique <products>/<product>/<name>

Each candidate string is normalized (lower, stripped of punctuation,
whitespace collapsed) and trailing noise tokens — dosage forms, routes,
common salt suffixes, dose strings like "5mg" — are peeled so
"Warfarin Sodium 2.5 mg Tab" reduces to "warfarin". Non-trailing
occurrences are preserved so compounds like "sodium chloride" are
indexed intact.

Usage
-----
    python etl/build_gazetteer.py --xml data/full_database.xml
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from parse_drugbank import (  # type: ignore[import-not-found]
    ALLOWED_GROUPS_DEFAULT,
    T,
    drug_groups,
    iter_drugs,
    primary_id,
)

# Trailing tokens we strip. Kept conservative and salt-aware: generic
# anions like "chloride" / "bromide" are NOT here, so "sodium chloride"
# and "potassium chloride" remain intact.
TRAILING_STOPWORDS: Set[str] = {
    # Dosage forms (singular + plural + common abbreviations)
    "tablet", "tablets", "tab", "tabs",
    "capsule", "capsules", "cap", "caps",
    "injection", "injections", "inj",
    "solution", "soln", "liquid", "syrup", "suspension", "susp",
    "cream", "ointment", "powder", "spray", "patch", "patches",
    "gel", "drops", "suppository", "suppositories",
    "film", "lozenge", "lozenges", "elixir", "emulsion", "kit",
    # Routes of administration
    "oral", "iv", "im", "sc", "topical", "subcutaneous",
    "intravenous", "intramuscular", "rectal", "nasal",
    "ophthalmic", "otic", "inhaled", "inhalation",
    # Common trailing salt modifiers
    "sodium", "potassium", "calcium", "magnesium",
    "hydrochloride", "hcl",
    "sulfate", "sulphate", "phosphate", "citrate", "acetate",
    "tartrate", "fumarate", "succinate", "maleate",
    "besylate", "mesylate", "tosylate",
    "hemihydrate", "hydrate", "monohydrate", "dihydrate",
    # Pharmacopeia / release qualifiers
    "usp", "bp", "ip", "eur", "ph",
    "er", "xr", "sr", "cr", "la", "xl", "ir",
}

# Units + numeric dose tokens. Peels "5mg", "10 ml", "100", "2.5".
_DOSE_RE = re.compile(
    r"^\d+(\.\d+)?(mg|mcg|ug|ml|l|g|kg|iu|u|units?|meq|mol|mmol|%)?$"
)
# Bare unit tokens left behind when the number and unit were separated by
# whitespace in the source ("5 mg" -> ["5", "mg"] after punct-strip).
_UNIT_TOKENS: Set[str] = {
    "mg", "mcg", "ug", "ml", "l", "g", "kg",
    "iu", "u", "unit", "units", "meq", "mol", "mmol",
}

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lower, strip punctuation, collapse whitespace, peel trailing noise."""
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    tokens = s.split()
    while tokens and (
        tokens[-1] in TRAILING_STOPWORDS
        or tokens[-1] in _UNIT_TOKENS
        or _DOSE_RE.match(tokens[-1])
    ):
        tokens.pop()
    return " ".join(tokens)


def collect_names(drug_elem) -> List[str]:
    """Return primary name, synonyms, and deduped product brand names."""
    out: List[str] = []

    name_el = drug_elem.find(T("name"))
    if name_el is not None and name_el.text:
        out.append(name_el.text.strip())

    syns = drug_elem.find(T("synonyms"))
    if syns is not None:
        for s in syns.findall(T("synonym")):
            if s.text and s.text.strip():
                out.append(s.text.strip())

    prods = drug_elem.find(T("products"))
    if prods is not None:
        seen: Set[str] = set()
        for p in prods.findall(T("product")):
            pn = p.find(T("name"))
            if pn is None or not pn.text:
                continue
            raw = pn.text.strip()
            key = raw.lower()
            if key and key not in seen:
                seen.add(key)
                out.append(raw)
    return out


def run(args: argparse.Namespace) -> None:
    allowed = None if args.include_all else ALLOWED_GROUPS_DEFAULT
    exact: Dict[str, str] = {}
    drugs_covered: Set[str] = set()
    per_drug_counts: List[int] = []
    n_seen = n_kept = 0

    print(f"→ Streaming drugs from {args.xml}", file=sys.stderr)
    for drug_elem in tqdm(iter_drugs(args.xml), unit="drug"):
        n_seen += 1
        dbid = primary_id(drug_elem)
        if not dbid:
            continue
        groups = drug_groups(drug_elem)
        if allowed is not None and not (groups & allowed):
            continue
        n_kept += 1

        norms: Set[str] = set()
        for raw in collect_names(drug_elem):
            n = normalize(raw)
            if not n:
                continue
            norms.add(n)
            exact.setdefault(n, dbid)
        if norms:
            drugs_covered.add(dbid)
            per_drug_counts.append(len(norms))

    # fuzzy list mirrors exact so a single normalized string resolves to
    # one canonical drug id. Two drugs can share a surface form (e.g.
    # "warfarin" appears for both DB00682 and DB14055 S-warfarin); the
    # first-seen wins here, same as the exact dict.
    fuzzy_pairs: List[Tuple[str, str]] = sorted(exact.items())

    exact_path = Path(args.exact_out).resolve()
    fuzzy_path = Path(args.fuzzy_out).resolve()
    exact_path.parent.mkdir(parents=True, exist_ok=True)
    fuzzy_path.parent.mkdir(parents=True, exist_ok=True)
    with exact_path.open("wb") as f:
        pickle.dump(exact, f)
    with fuzzy_path.open("wb") as f:
        pickle.dump(fuzzy_pairs, f)

    avg = (sum(per_drug_counts) / len(per_drug_counts)) if per_drug_counts else 0.0

    print("\n" + "=" * 60, file=sys.stderr)
    print("GAZETTEER BUILD SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Drugs seen in XML:         {n_seen:,}", file=sys.stderr)
    print(f"Drugs kept:                {n_kept:,}", file=sys.stderr)
    print(f"Drugs covered:             {len(drugs_covered):,}", file=sys.stderr)
    print(f"Unique normalized strings: {len(exact):,}", file=sys.stderr)
    print(f"Fuzzy (norm, id) pairs:    {len(fuzzy_pairs):,}", file=sys.stderr)
    print(f"Avg names per covered drug: {avg:.1f}", file=sys.stderr)
    print(f"\nWrote {exact_path}", file=sys.stderr)
    print(f"Wrote {fuzzy_path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", required=True, help="Path to unzipped DrugBank XML")
    p.add_argument(
        "--exact-out",
        default="./gazetteer.pkl",
        help="Output path for the exact-lookup dict pickle",
    )
    p.add_argument(
        "--fuzzy-out",
        default="./gazetteer_fuzzy.pkl",
        help="Output path for the fuzzy (name, drug_id) list pickle",
    )
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Keep all drug groups (default: approved+investigational only)",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
