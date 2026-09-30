import json
import pytest
from mgvqa.metrics import answer_metrics, is_answer_correct
from mgvqa.orchestrator.summary import generate_summary


def records():
    return [
        {'question_type': 'count', 'status': 'answered', 'is_correct': True},
        {'question_type': 'count', 'status': 'answered', 'is_correct': False},
        {'question_type': 'count', 'status': 'fail', 'fail_reason': 'Maximum VLM steps exceeded (35 steps).', 'is_correct': True},
        {'question_type': 'count', 'status': 'perception_loop_failure'},
        {'question_type': 'count', 'status': 'error', 'fail_reason': 'API timeout'},
        {'question_type': 'count', 'status': 'skipped'},
        {'question_type': 'find', 'status': 'fail', 'fail_reason': 'Empty answer'},
    ]


@pytest.mark.parametrize('scale', [1, 100])
def test_answer_only_denominators(scale):
    stats = answer_metrics(records(), scale=scale)
    count = stats['per_type']['count']
    assert count['accuracy'] == 0.5 * scale
    assert count['answered'] == 2 and count['correct'] == 1
    assert count['non_answer'] == 4
    assert stats['per_type']['find']['answered'] == 0
    assert len(stats['non_answer_failures_by_reason']) == 4
    assert not is_answer_correct(records()[2])


def test_merged_summary_excludes_all_nonanswers(tmp_path):
    generate_summary(str(tmp_path), records())
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['accuracy'] == 0.5
    assert summary['per_type']['count']['accuracy'] == 0.5
    assert summary['n_correct'] == 1
    assert summary['n_episodes_failed'] == 3
    assert summary['n_episodes_infra'] == 1
    assert summary['n_episodes_max_steps'] == 1
    assert summary['n_episodes_perception_loop_failure'] == 1
    assert 'accuracy_attempted' not in summary
