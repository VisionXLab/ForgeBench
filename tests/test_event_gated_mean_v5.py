"""Behavioral regressions for the single sample-level coverage gate."""
from copy import deepcopy

import pytest

from scoring.aggregate import aggregate_sample_results, compute_sample_ranking_score
from scoring.migration import rebuild_cached_result
from scoring.per_sample import score_sample
from scoring.policy import CONFIG


def row(quality=80, coverage=50, task='a', evidence=None):
    return {'task_id': task, 'operator_evidence': evidence,
            'scored': score_sample({a: quality for a in CONFIG['headline_axes']},
                                   observable_event_coverage=coverage, operator_evidence=evidence)}


@pytest.mark.parametrize('coverage,expected', [(0, 0), (25, 20), (50, 40), (100, 80)])
def test_coverage_applies_once_and_does_not_flatten_axes(coverage, expected):
    result = row(coverage=coverage)
    assert compute_sample_ranking_score(result) == expected
    assert all(x == 80 for x in result['scored']['axis_scores'].values())
    assert result['scored']['application_usefulness_score'] == 80
    assert aggregate_sample_results([result])['ranking_score'] == expected


def test_average_sample_products_not_product_of_model_means():
    rows = [row(100, 0, 'a'), row(20, 100, 'b')]
    result = aggregate_sample_results(rows)
    assert result['ranking_score'] == 10  # not mean(100,20)*mean(0,1)=30


def test_missing_coverage_is_incomplete_not_full_credit():
    result = row(coverage=None)
    assert compute_sample_ranking_score(result) is None
    assert aggregate_sample_results([result])['ranking_publishable'] is False


@pytest.mark.parametrize('present', ['false', 0, None])
def test_invalid_event_check_is_not_treated_as_observed_failure(present):
    result = row(coverage=None)
    result['application_usefulness_details'] = {'required_event_checks': [{'present': present}]}
    assert compute_sample_ranking_score(result) is None
    assert compute_sample_ranking_score(rebuild_cached_result(result)) is None


def test_coverage_can_be_recovered_from_valid_event_checks():
    result = row(coverage=None)
    result['application_usefulness_details'] = {'required_event_checks': [{'present': True}, {'present': False}]}
    assert compute_sample_ranking_score(result) == 40
    assert compute_sample_ranking_score(rebuild_cached_result(result)) == 40


def test_operator_cap_precedes_coverage_and_is_not_reapplied_after_gate():
    evidence = {'operators': {'temporal_break': {'used_for_axis_cap': True,
                'confidence': 0.9, 'validity': 'valid', 'abrupt_transition': True}}}
    result = row(coverage=50, evidence=evidence)
    assert compute_sample_ranking_score(result) == pytest.approx((5 * 80 + 35) / 6 * 0.5)


def test_old_event_capped_cache_rebuilds_from_original_axes_idempotently():
    result = row(coverage=50)
    result['scored']['axis_scores'] = {a: 50 for a in CONFIG['technical_axes']}
    result['scored']['application_usefulness_score'] = 50
    result['scored']['scoring_policy'] = {'version': 'forge-video-6axis-capped-mean-v4.1'}
    original = deepcopy(result)
    assert compute_sample_ranking_score(result) is None
    rebuilt = rebuild_cached_result(result)
    assert compute_sample_ranking_score(rebuilt) == 40
    assert rebuild_cached_result(rebuilt) == rebuilt
    assert original == result
