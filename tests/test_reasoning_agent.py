"""Reasoning agent tests.

Two tiers:

* **Offline** — evidence formatters, token-budget truncation, and the
  invariant #7 short-circuit. No Groq calls; runs in <1s.
* **Live Groq** — five mode-specific canned queries plus a revision
  loop. Each is individually ``@pytest.mark.skip``'d so a plain
  ``pytest`` run does not burn quota; remove the skip on the one you
  want to run, or invoke via ``-k``.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from agent.nodes.reasoning_agent import (
    MAX_EVIDENCE_TOKENS,
    _collect_lines,
    _count_tokens,
    _insufficient_output,
    _pack_enzyme_row,
    _pack_evidence,
    _pack_pair_row,
    _pack_vector_chunk,
    reason,
    reasoning_agent,
)
from agent.schemas import ReasoningOutput

load_dotenv()


# ---------------------------------------------------------------- #
# Offline: evidence formatters
# ---------------------------------------------------------------- #
def test_pack_pair_row_direct() -> None:
    row = {
        "drug_a_id": "DB00682",
        "drug_a_name": "Warfarin",
        "drug_b_id": "DB00945",
        "drug_b_name": "Aspirin",
        "query_type": "direct",
        "severity": "Major",
        "description": "Increased bleeding risk when combined.",
    }
    line = _pack_pair_row(row)
    assert "Warfarin (DB00682)" in line
    assert "Aspirin (DB00945)" in line
    assert "[direct]" in line
    assert "severity=Major" in line
    assert "Increased bleeding risk" in line


def test_pack_pair_row_multi_hop() -> None:
    row = {
        "drug_a_id": "DB00001",
        "drug_a_name": "DrugA",
        "drug_b_id": "DB00002",
        "drug_b_name": "DrugB",
        "query_type": "multi_hop",
        "enzyme_name": "CYP3A4",
        "a_actions": ["inhibitor"],
        "b_actions": ["substrate"],
    }
    line = _pack_pair_row(row)
    assert "multi_hop via CYP3A4" in line
    assert "DrugA inhibitor" in line
    assert "DrugB substrate" in line


def test_pack_enzyme_row() -> None:
    row = {
        "drug_id": "DB00001",
        "drug_name": "Ketoconazole",
        "enzyme_name": "CYP3A4",
        "actions": ["inhibitor"],
    }
    line = _pack_enzyme_row(row)
    assert "Ketoconazole (DB00001)" in line
    assert "enzyme_filter" in line
    assert "inhibitor" in line
    assert "CYP3A4" in line


def test_pack_vector_chunk() -> None:
    row = {
        "drug_id": "DB13928",
        "drug_name": "Semaglutide",
        "collection": "mechanisms",
        "text": "GLP-1 receptor agonist.",
    }
    line = _pack_vector_chunk(row)
    assert "Semaglutide (DB13928)" in line
    assert "[mechanisms]" in line
    assert "GLP-1 receptor agonist" in line


# ---------------------------------------------------------------- #
# Offline: _collect_lines mode gating
# ---------------------------------------------------------------- #
def test_collect_lines_ddi_check_graph_only() -> None:
    fused = [
        {
            "drug_id": "DB00682",
            "graph": [
                {
                    "drug_a_id": "DB00682",
                    "drug_a_name": "Warfarin",
                    "drug_b_id": "DB00945",
                    "drug_b_name": "Aspirin",
                    "query_type": "direct",
                    "severity": "Major",
                    "description": "bleeding",
                }
            ],
            "vector": [
                {
                    "drug_id": "DB00682",
                    "drug_name": "Warfarin",
                    "collection": "descriptions",
                    "text": "ignored in ddi_check",
                }
            ],
        }
    ]
    g, v = _collect_lines(fused, "ddi_check")
    assert len(g) == 1
    assert v == []


def test_collect_lines_describe_vector_only() -> None:
    fused = [
        {
            "drug_id": "DB13928",
            "graph": [
                {
                    "drug_a_id": "X",
                    "drug_b_id": "Y",
                    "query_type": "direct",
                    "severity": "Minor",
                    "description": "ignored",
                }
            ],
            "vector": [
                {
                    "drug_id": "DB13928",
                    "drug_name": "Semaglutide",
                    "collection": "mechanisms",
                    "text": "GLP-1 agonist",
                }
            ],
        }
    ]
    g, v = _collect_lines(fused, "describe")
    assert g == []
    assert len(v) == 1


def test_collect_lines_hybrid_uses_both() -> None:
    fused = [
        {
            "drug_id": "DB00001",
            "graph": [
                {
                    "drug_id": "DB00001",
                    "drug_name": "DrugA",
                    "enzyme_name": "CYP3A4",
                    "actions": ["inhibitor"],
                    "query_type": "enzyme_filter",
                }
            ],
            "vector": [
                {
                    "drug_id": "DB00001",
                    "drug_name": "DrugA",
                    "collection": "indications",
                    "text": "text",
                }
            ],
        }
    ]
    g, v = _collect_lines(fused, "hybrid")
    assert len(g) == 1
    assert len(v) == 1


def test_collect_lines_dedupes_shared_pair_rows() -> None:
    """Pairwise expansion emits the same row twice (once per endpoint);
    the formatter must not double-print it."""
    shared_row = {
        "drug_a_id": "DB00682",
        "drug_a_name": "Warfarin",
        "drug_b_id": "DB00945",
        "drug_b_name": "Aspirin",
        "query_type": "direct",
        "severity": "Major",
        "description": "bleeding",
    }
    fused = [
        {"drug_id": "DB00682", "graph": [shared_row], "vector": []},
        {"drug_id": "DB00945", "graph": [shared_row], "vector": []},
    ]
    g, _ = _collect_lines(fused, "ddi_check")
    assert len(g) == 1


# ---------------------------------------------------------------- #
# Offline: _pack_evidence budgeting
# ---------------------------------------------------------------- #
def test_pack_evidence_empty_fused() -> None:
    assert _pack_evidence({"fused_results": [], "mode": "ddi_check"}) == (
        "<no evidence available>"
    )


def test_pack_evidence_no_matching_lines_returns_sentinel() -> None:
    """ddi_check on fused_results with only vector chunks → no usable rows."""
    fused = [
        {
            "drug_id": "DB13928",
            "graph": [],
            "vector": [
                {
                    "drug_id": "DB13928",
                    "drug_name": "Semaglutide",
                    "collection": "mechanisms",
                    "text": "GLP-1 agonist",
                }
            ],
        }
    ]
    out = _pack_evidence({"fused_results": fused, "mode": "ddi_check"})
    assert out == "<no evidence available>"


def test_pack_evidence_renders_both_blocks_for_hybrid() -> None:
    fused = [
        {
            "drug_id": "DB00001",
            "graph": [
                {
                    "drug_id": "DB00001",
                    "drug_name": "Ketoconazole",
                    "enzyme_name": "CYP3A4",
                    "actions": ["inhibitor"],
                    "query_type": "enzyme_filter",
                }
            ],
            "vector": [
                {
                    "drug_id": "DB00001",
                    "drug_name": "Ketoconazole",
                    "collection": "indications",
                    "text": "antifungal",
                }
            ],
        }
    ]
    out = _pack_evidence(
        {
            "fused_results": fused,
            "mode": "hybrid",
            "semantic_constraint": "statin",
        }
    )
    assert "Semantic constraint: statin" in out
    assert "INTERACTION / ENZYME EVIDENCE:" in out
    assert "DRUG TEXT EVIDENCE:" in out


def test_pack_evidence_truncates_oversized_block() -> None:
    """Fabricate 200 pair rows, each ~30 tokens of description — the total
    well exceeds MAX_EVIDENCE_TOKENS (2500) and the footer must appear."""
    big_desc = "lorem ipsum dolor sit amet " * 5  # ~25 tokens each
    fused: list[dict[str, Any]] = []
    for i in range(200):
        row = {
            "drug_a_id": f"DB{i:05d}",
            "drug_a_name": f"DrugA{i}",
            "drug_b_id": f"DC{i:05d}",
            "drug_b_name": f"DrugB{i}",
            "query_type": "direct",
            "severity": "Major",
            "description": big_desc,
        }
        fused.append({"drug_id": f"DB{i:05d}", "graph": [row], "vector": []})

    out = _pack_evidence({"fused_results": fused, "mode": "ddi_check"})
    assert "additional entries truncated" in out
    assert _count_tokens(out) <= MAX_EVIDENCE_TOKENS + 50


def test_pack_evidence_respects_total_budget() -> None:
    """Even with graph + vector + header, total evidence must fit in budget."""
    big_desc = "lorem ipsum dolor sit amet " * 5
    fused: list[dict[str, Any]] = []
    for i in range(150):
        fused.append(
            {
                "drug_id": f"DB{i:05d}",
                "graph": [
                    {
                        "drug_a_id": f"DB{i:05d}",
                        "drug_a_name": f"A{i}",
                        "drug_b_id": f"DC{i:05d}",
                        "drug_b_name": f"B{i}",
                        "query_type": "direct",
                        "severity": "Minor",
                        "description": big_desc,
                    }
                ],
                "vector": [
                    {
                        "drug_id": f"DB{i:05d}",
                        "drug_name": f"A{i}",
                        "collection": "indications",
                        "text": big_desc,
                    }
                ],
            }
        )

    out = _pack_evidence(
        {
            "fused_results": fused,
            "mode": "hybrid",
            "semantic_constraint": "statin",
        }
    )
    assert _count_tokens(out) <= MAX_EVIDENCE_TOKENS + 50


# ---------------------------------------------------------------- #
# Offline: invariant #7 short-circuit (no Groq)
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mode", ["ddi_check", "polypharmacy", "alternatives", "hybrid"]
)
def test_reason_short_circuits_on_empty_evidence(mode: str) -> None:
    state: dict[str, Any] = {
        "query": "whatever",
        "mode": mode,
        "resolved_drugs": [{"drug_id": "DB00682"}, {"drug_id": "DB00945"}],
        "fused_results": [],
    }
    out = reason(state)
    assert isinstance(out, ReasoningOutput)
    assert out.insufficient_evidence is True
    assert out.severity == "unknown"
    assert out.confidence == 0.0
    assert set(out.sources) == {"DB00682", "DB00945"}


def test_reason_short_circuits_describe_with_na_severity() -> None:
    state: dict[str, Any] = {
        "query": "what is x",
        "mode": "describe",
        "resolved_drugs": [{"drug_id": "DB13928"}],
        "fused_results": [],
    }
    out = reason(state)
    assert out.insufficient_evidence is True
    assert out.severity == "n/a"
    assert out.sources == ["DB13928"]


def test_reason_short_circuits_on_unknown_mode() -> None:
    state: dict[str, Any] = {
        "query": "x",
        "mode": None,
        "resolved_drugs": [],
        "fused_results": [],
    }
    out = reason(state)
    assert out.insufficient_evidence is True


def test_insufficient_output_describe_shape() -> None:
    out = _insufficient_output("describe", ["DB13928"])
    assert out.mode == "describe"
    assert out.severity == "n/a"
    assert out.summary.startswith("No descriptive chunks")


def test_insufficient_output_ddi_shape() -> None:
    out = _insufficient_output("ddi_check", ["DB00682"])
    assert out.mode == "ddi_check"
    assert out.severity == "unknown"


def test_reasoning_agent_node_wraps_dict() -> None:
    state: dict[str, Any] = {
        "query": "x",
        "mode": "ddi_check",
        "resolved_drugs": [{"drug_id": "DB00682"}],
        "fused_results": [],
    }
    update = reasoning_agent(state)
    assert "reasoning_output" in update
    assert update["reasoning_output"]["insufficient_evidence"] is True  # type: ignore[index]


# ---------------------------------------------------------------- #
# Offline: revision path prepends prior draft + feedback
# ---------------------------------------------------------------- #
def test_revision_prefix_included_when_critic_feedback_present() -> None:
    """_format_user_message must prepend REVISION_INSTRUCTION_TEMPLATE
    content when critic_feedback + prior reasoning_output are both set."""
    from agent.nodes.reasoning_agent import _format_user_message

    state: dict[str, Any] = {
        "query": "is warfarin safe with aspirin",
        "mode": "ddi_check",
        "fused_results": [
            {
                "drug_id": "DB00682",
                "graph": [
                    {
                        "drug_a_id": "DB00682",
                        "drug_a_name": "Warfarin",
                        "drug_b_id": "DB00945",
                        "drug_b_name": "Aspirin",
                        "query_type": "direct",
                        "severity": "Major",
                        "description": "bleeding",
                    }
                ],
                "vector": [],
            }
        ],
        "critic_feedback": "Mechanism was missing — explain the PK basis.",
        "reasoning_output": {
            "mode": "ddi_check",
            "severity": "Major",
            "mechanism": "",
            "recommendation": "avoid",
            "confidence": 0.5,
            "insufficient_evidence": False,
            "sources": ["DB00682", "DB00945"],
        },
    }
    msg = _format_user_message(state)
    assert "Your prior response had these issues" in msg
    assert "Mechanism was missing" in msg
    assert "Prior draft" in msg
    assert "User question:" in msg
    assert "Evidence:" in msg


# ---------------------------------------------------------------- #
# Live Groq gate fixture + per-test skip
# ---------------------------------------------------------------- #
@pytest.fixture
def _require_groq_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


def _ddi_state() -> dict[str, Any]:
    return {
        "query": "Is warfarin safe with aspirin?",
        "mode": "ddi_check",
        "resolved_drugs": [{"drug_id": "DB00682"}, {"drug_id": "DB00945"}],
        "fused_results": [
            {
                "drug_id": "DB00682",
                "graph": [
                    {
                        "drug_a_id": "DB00682",
                        "drug_a_name": "Warfarin",
                        "drug_b_id": "DB00945",
                        "drug_b_name": "Aspirin",
                        "query_type": "direct",
                        "severity": "Major",
                        "description": (
                            "Aspirin may increase the anticoagulant "
                            "activity of Warfarin, increasing bleeding risk."
                        ),
                    }
                ],
                "vector": [],
            }
        ],
    }


def _describe_state() -> dict[str, Any]:
    return {
        "query": "What is semaglutide?",
        "mode": "describe",
        "resolved_drugs": [{"drug_id": "DB13928"}],
        "fused_results": [
            {
                "drug_id": "DB13928",
                "graph": [],
                "vector": [
                    {
                        "drug_id": "DB13928",
                        "drug_name": "Semaglutide",
                        "collection": "descriptions",
                        "text": (
                            "Semaglutide is a GLP-1 receptor agonist used "
                            "for type 2 diabetes and chronic weight "
                            "management."
                        ),
                    },
                    {
                        "drug_id": "DB13928",
                        "drug_name": "Semaglutide",
                        "collection": "mechanisms",
                        "text": (
                            "Binds the GLP-1 receptor, stimulating "
                            "glucose-dependent insulin secretion and "
                            "suppressing glucagon release."
                        ),
                    },
                ],
            }
        ],
    }


def _alternatives_state() -> dict[str, Any]:
    return {
        "query": "What anticoagulant is safe in HIT?",
        "mode": "alternatives",
        "semantic_constraint": "anticoagulant safe in HIT",
        "fused_results": [
            {
                "drug_id": "DB00001",
                "graph": [],
                "vector": [
                    {
                        "drug_id": "DB00001",
                        "drug_name": "Lepirudin",
                        "collection": "indications",
                        "text": (
                            "Lepirudin is indicated for anticoagulation "
                            "in patients with heparin-induced "
                            "thrombocytopenia (HIT)."
                        ),
                    }
                ],
            },
            {
                "drug_id": "DB00278",
                "graph": [],
                "vector": [
                    {
                        "drug_id": "DB00278",
                        "drug_name": "Argatroban",
                        "collection": "indications",
                        "text": (
                            "Argatroban is a direct thrombin inhibitor "
                            "indicated for prophylaxis or treatment of "
                            "thrombosis in HIT."
                        ),
                    }
                ],
            },
            {
                "drug_id": "DB00006",
                "graph": [],
                "vector": [
                    {
                        "drug_id": "DB00006",
                        "drug_name": "Bivalirudin",
                        "collection": "indications",
                        "text": (
                            "Bivalirudin is a direct thrombin inhibitor "
                            "used for anticoagulation during PCI, "
                            "including patients with HIT."
                        ),
                    }
                ],
            },
        ],
    }


def _hybrid_state() -> dict[str, Any]:
    return {
        "query": "Which CYP3A4 inhibitors interact with statins?",
        "mode": "hybrid",
        "semantic_constraint": "statin",
        "fused_results": [
            {
                "drug_id": "DB01211",
                "graph": [
                    {
                        "drug_id": "DB01211",
                        "drug_name": "Clarithromycin",
                        "enzyme_name": "CYP3A4",
                        "actions": ["inhibitor"],
                        "query_type": "enzyme_filter",
                    }
                ],
                "vector": [
                    {
                        "drug_id": "DB01211",
                        "drug_name": "Clarithromycin",
                        "collection": "indications",
                        "text": (
                            "Clarithromycin is a macrolide antibiotic; "
                            "concomitant use with simvastatin is "
                            "contraindicated due to CYP3A4 inhibition."
                        ),
                    }
                ],
            },
            {
                "drug_id": "DB01167",
                "graph": [
                    {
                        "drug_id": "DB01167",
                        "drug_name": "Itraconazole",
                        "enzyme_name": "CYP3A4",
                        "actions": ["inhibitor"],
                        "query_type": "enzyme_filter",
                    }
                ],
                "vector": [
                    {
                        "drug_id": "DB01167",
                        "drug_name": "Itraconazole",
                        "collection": "indications",
                        "text": (
                            "Itraconazole is a triazole antifungal that "
                            "potently inhibits CYP3A4; co-administration "
                            "with simvastatin markedly raises statin "
                            "exposure and rhabdomyolysis risk."
                        ),
                    }
                ],
            },
        ],
    }


def _polypharmacy_state() -> dict[str, Any]:
    pair = lambda a, an, b, bn, sev, desc: {  # noqa: E731
        "drug_a_id": a,
        "drug_a_name": an,
        "drug_b_id": b,
        "drug_b_name": bn,
        "query_type": "direct",
        "severity": sev,
        "description": desc,
    }
    return {
        "query": "Patient on warfarin, aspirin, clopidogrel, omeprazole, metformin — concerns?",
        "mode": "polypharmacy",
        "resolved_drugs": [
            {"drug_id": "DB00682"},
            {"drug_id": "DB00945"},
            {"drug_id": "DB00758"},
            {"drug_id": "DB00338"},
            {"drug_id": "DB00331"},
        ],
        "fused_results": [
            {
                "drug_id": "DB00682",
                "graph": [
                    pair(
                        "DB00682",
                        "Warfarin",
                        "DB00945",
                        "Aspirin",
                        "Major",
                        "Increased bleeding risk when combined.",
                    ),
                    pair(
                        "DB00682",
                        "Warfarin",
                        "DB00758",
                        "Clopidogrel",
                        "Major",
                        (
                            "Additive antiplatelet/anticoagulant "
                            "effects increase bleeding."
                        ),
                    ),
                ],
                "vector": [],
            },
            {
                "drug_id": "DB00758",
                "graph": [
                    pair(
                        "DB00758",
                        "Clopidogrel",
                        "DB00338",
                        "Omeprazole",
                        "Moderate",
                        (
                            "Omeprazole inhibits CYP2C19, reducing "
                            "clopidogrel active metabolite formation."
                        ),
                    ),
                ],
                "vector": [],
            },
        ],
    }


# ---------------------------------------------------------------- #
# Live Groq tests — SKIPPED BY DEFAULT. Remove the skip to run.
# ---------------------------------------------------------------- #
@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_ddi_check_warfarin_aspirin(_require_groq_key: None) -> None:
    out = reason(_ddi_state())
    import json as _json
    print("\n----- ReasoningOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("-------------------------------")
    assert out.mode == "ddi_check"
    assert out.severity in ("Major", "Moderate")
    assert not out.insufficient_evidence
    assert len(out.interacting_pairs) == 1
    assert out.mechanism
    assert out.recommendation
    assert set(out.sources) >= {"DB00682", "DB00945"}


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_describe_semaglutide(_require_groq_key: None) -> None:
    out = reason(_describe_state())
    import json as _json
    print("\n----- ReasoningOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("--------------------------------")
    assert out.mode == "describe"
    assert out.severity == "n/a"
    assert not out.insufficient_evidence
    assert out.summary
    assert "GLP" in out.summary or "GLP" in out.mechanism
    assert out.sources == ["DB13928"]


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_alternatives_hit(_require_groq_key: None) -> None:
    out = reason(_alternatives_state())
    import json as _json
    print("\n----- ReasoningOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("--------------------------------")
    assert out.mode == "alternatives"
    assert out.severity == "n/a"
    assert 2 <= len(out.candidates) <= 5
    cand_ids = {c["drug_id"] for c in out.candidates}
    assert cand_ids <= {"DB00001", "DB00278", "DB00006"}


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_hybrid_cyp3a4_statins(_require_groq_key: None) -> None:
    out = reason(_hybrid_state())
    import json as _json
    print("\n----- ReasoningOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("--------------------------------")
    assert out.mode == "hybrid"
    assert 2 <= len(out.candidates) <= 5
    cand_ids = {c["drug_id"] for c in out.candidates}
    assert cand_ids <= {"DB01211", "DB01167"}
    assert "CYP3A4" in out.mechanism


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_polypharmacy_five_drug(_require_groq_key: None) -> None:
    out = reason(_polypharmacy_state())
    import json as _json
    print("\n----- ReasoningOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("--------------------------------")
    assert out.mode == "polypharmacy"
    assert out.severity in ("Major", "Moderate")
    assert len(out.interacting_pairs) >= 2
    # Severity ordering: Major entries precede Moderate entries.
    severities = [p["severity"] for p in out.interacting_pairs]
    rank = {"Major": 0, "Moderate": 1, "Minor": 2, "unknown": 3}
    assert severities == sorted(severities, key=lambda s: rank.get(s, 99))


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_revision_patches_prior_draft(_require_groq_key: None) -> None:
    """Pipe a deliberately-thin prior draft + critic feedback; the
    patched draft should address the feedback without re-deriving
    severity from scratch."""
    state = _ddi_state()
    state["critic_feedback"] = (
        "Mechanism field is empty. Name the pharmacodynamic "
        "basis (platelet inhibition + vitamin-K antagonism)."
    )
    state["reasoning_output"] = {
        "mode": "ddi_check",
        "severity": "Major",
        "interacting_pairs": [
            {
                "drug_a": "Warfarin (DB00682)",
                "drug_b": "Aspirin (DB00945)",
                "severity": "Major",
                "rationale": "bleeding risk",
            }
        ],
        "mechanism": "",
        "recommendation": "avoid",
        "confidence": 0.7,
        "insufficient_evidence": False,
        "sources": ["DB00682", "DB00945"],
    }
    out = reason(state)
    import json as _json
    print("\n----- ReasoningOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("--------------------------------")
    assert out.severity == "Major"  # preserved from prior draft
    assert out.mechanism  # now populated
    lowered = out.mechanism.lower()
    assert "platelet" in lowered or "vitamin" in lowered or "coagul" in lowered


# ---------------------------------------------------------------- #
# Offline: LLM call is mocked so we exercise reason()'s parse path
# without burning Groq quota.
# ---------------------------------------------------------------- #
def test_reason_parses_mocked_llm_response() -> None:
    state = _ddi_state()
    canned = (
        '{"mode": "ddi_check", "severity": "Major", '
        '"interacting_pairs": [{"drug_a": "Warfarin (DB00682)", '
        '"drug_b": "Aspirin (DB00945)", "severity": "Major", '
        '"rationale": "bleeding"}], "mechanism": "antiplatelet + '
        'anticoagulant additive effect", "recommendation": "avoid '
        'combination", "confidence": 0.9, "insufficient_evidence": '
        'false, "sources": ["DB00682", "DB00945"]}'
    )
    with patch(
        "agent.nodes.reasoning_agent._invoke", return_value=canned
    ):
        out = reason(state)
    assert out.mode == "ddi_check"
    assert out.severity == "Major"
    assert out.confidence == 0.9
    assert not out.insufficient_evidence
