from eval.axis_registry import (
    APPLICATION_USEFULNESS,
    GEOMETRIC_INTEGRITY,
    INDUSTRIAL_LOGIC_AND_FACT_ALIGNMENT,
    PHYSICAL_PLAUSIBILITY,
    REFERENCE_AND_MOTION_FIDELITY,
    TEMPORAL_CONSISTENCY,
)
from eval.run_eval import _sample_scoring_validity
import pytest
from scoring.policy import CONFIG


def test_transformed_detail_score_names_are_not_false_parse_failures():
    scores = {
        INDUSTRIAL_LOGIC_AND_FACT_ALIGNMENT: 60,
        GEOMETRIC_INTEGRITY: 70,
        PHYSICAL_PLAUSIBILITY: 65,
        TEMPORAL_CONSISTENCY: 55,
        REFERENCE_AND_MOTION_FIDELITY: 50,
        APPLICATION_USEFULNESS: 45,
    }
    details = {
        TEMPORAL_CONSISTENCY: {
            "temporal_consistency_score": 55,
            "llm_parse_valid": True,
            "raw_response": '{"score":55}',
        },
        REFERENCE_AND_MOTION_FIDELITY: {
            "reference_and_motion_fidelity_score": 50,
            "llm_parse_valid": True,
            "raw_response": '{"score":50}',
        },
    }
    validity = _sample_scoring_validity(scores, details)
    assert validity["complete_required_axes"] is True
    assert validity["invalid_judge_outputs"] == []


@pytest.mark.parametrize("value", [None, True, "75", float("nan"), float("inf"), -1, 101])
def test_invalid_numbers_cannot_mark_sample_complete(value):
    scores = dict.fromkeys(CONFIG["headline_axes"], 75)
    scores[TEMPORAL_CONSISTENCY] = value
    validity = _sample_scoring_validity(scores, {})
    assert validity["complete_required_axes"] is False
    assert validity["invalid_axis_scores"] == [TEMPORAL_CONSISTENCY]


def test_failed_judge_cannot_be_hidden_by_numeric_fallback():
    scores = dict.fromkeys(CONFIG["headline_axes"], 75)
    validity = _sample_scoring_validity(scores, {
        TEMPORAL_CONSISTENCY: {"llm_parse_valid": False, "error": "timeout"},
    })
    assert validity["complete_required_axes"] is False
    assert validity["invalid_judge_outputs"] == [TEMPORAL_CONSISTENCY]
