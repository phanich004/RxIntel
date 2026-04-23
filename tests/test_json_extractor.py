"""Unit tests for ``agent.nodes.reasoning_agent._extract_json``.

The extractor is the safety net between Claude's free-form output and
the strict ``json.loads`` the reasoning/critic agents do. We care about
three classes of input: clean JSON, fence-wrapped JSON, and
prose-wrapped JSON. The nested and string-with-braces cases exist to
guard the balanced-brace scanner against off-by-one breakage.
"""

from __future__ import annotations

import json

import pytest

from agent.nodes.reasoning_agent import _extract_json


def test_plain_json() -> None:
    assert _extract_json('{"foo": 1}') == '{"foo": 1}'


def test_fenced_json() -> None:
    assert _extract_json('```json\n{"foo": 1}\n```') == '{"foo": 1}'
    assert _extract_json('```\n{"foo": 1}\n```') == '{"foo": 1}'


def test_prose_prefix() -> None:
    result = _extract_json('Here is the analysis:\n\n{"foo": 1}')
    assert json.loads(result) == {"foo": 1}


def test_prose_suffix() -> None:
    result = _extract_json('{"foo": 1}\n\nNote: draft.')
    assert json.loads(result) == {"foo": 1}


def test_nested_json() -> None:
    result = _extract_json('{"outer": {"inner": 1}}')
    assert json.loads(result) == {"outer": {"inner": 1}}


def test_json_with_string_containing_braces() -> None:
    """The string ``"{not json}"`` inside must not confuse the scanner."""
    result = _extract_json('{"msg": "text with {braces} inside"}')
    assert json.loads(result) == {"msg": "text with {braces} inside"}


def test_prose_with_false_brace_then_real_json() -> None:
    """A ``{example}`` in prose must not derail extraction of the real object."""
    result = _extract_json(
        'Given context {example}, the answer is:\n{"foo": 1}'
    )
    assert json.loads(result) == {"foo": 1}


def test_empty_string_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        json.loads(_extract_json(""))


def test_no_json_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        json.loads(_extract_json("This is just prose with no object."))
