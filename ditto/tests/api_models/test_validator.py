"""Wire-shape tests for the validator ScoreReport / CaseScore models.

Focus: the per-case breakdown must round-trip the DittoBench Go scorer's
``CaseScore`` for *both* case families — in particular a memory case (where the
signal is ``kind=memory`` + ``correct`` + ``score``, not ``tool_score``), and
the Go scorer's null/absent list fields.
"""

from __future__ import annotations

from ditto.api_models.validator import CaseScore, ScoreReport


def test_tool_case_round_trips() -> None:
    cs = CaseScore.model_validate(
        {
            "case_id": "web_search-1-0001",
            "category": "web_search",
            "kind": "tool",
            "score": 0.9,
            "tool_score": 1.0,
            "quality": 0.8,
            "latency_ms": 42,
            "called": ["search_web"],
            "expected": ["search_web"],
        }
    )
    assert cs.kind == "tool"
    assert cs.score == 0.9
    assert cs.correct is False  # default, absent on tool cases


def test_memory_case_preserves_signal() -> None:
    # As the Go scorer emits a memory case: kind=memory, correct verdict, the
    # meaningful number in `score`, tool_score 0, and expected as [].
    cs = CaseScore.model_validate(
        {
            "case_id": "m-0003",
            "category": "multi-session",
            "kind": "memory",
            "score": 1.0,
            "tool_score": 0.0,
            "correct": True,
            "latency_ms": 20,
            "called": [],
            "expected": [],
        }
    )
    assert cs.kind == "memory"
    assert cs.correct is True
    assert cs.score == 1.0  # not lost despite tool_score == 0


def test_null_list_fields_coerced_to_empty() -> None:
    # A nil Go slice serializes to JSON null; the model must accept it.
    cs = CaseScore.model_validate(
        {
            "case_id": "m-0004",
            "category": "memory",
            "kind": "memory",
            "score": 0.0,
            "tool_score": 0.0,
            "latency_ms": 5,
            "called": None,
            "expected": None,
            "notes": None,
        }
    )
    assert cs.called == []
    assert cs.expected == []
    assert cs.notes == []


def test_scorereport_with_mixed_per_case() -> None:
    report = ScoreReport.model_validate(
        {
            "run_id": "r1",
            "seed": 8675309,
            "composite": 0.6,  # 0.6*1.0 + 0.4*0.0
            "tool_mean": 1.0,
            "memory_mean": 0.0,
            "median_ms": 30,
            "n": 2,
            "generated_at": "2026-06-30T00:00:00Z",
            "per_case": [
                {
                    "case_id": "t1",
                    "category": "web_search",
                    "kind": "tool",
                    "score": 1.0,
                    "tool_score": 1.0,
                    "quality": 1.0,
                    "latency_ms": 10,
                    "called": ["search_web"],
                    "expected": ["search_web"],
                },
                {
                    "case_id": "m1",
                    "category": "multi-session",
                    "kind": "memory",
                    "score": 0.0,
                    "tool_score": 0.0,
                    "correct": False,
                    "latency_ms": 50,
                    "called": [],
                    "expected": [],
                },
            ],
        }
    )
    assert report.seed == 8675309
    assert [c.kind for c in report.per_case] == ["tool", "memory"]
