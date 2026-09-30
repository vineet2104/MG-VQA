"""Published prompt contract: six templates, no search hints or privileged fields."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mgvqa.prompts import build_user_prompt, format_previous_tool_response, load_system_prompt
from mgvqa.tool_registry import canonical_name, normalize_planner_call, public_name


@pytest.mark.parametrize('mode', ['direct', 'perception', 'agentic'])
@pytest.mark.parametrize('task', ['count', 'find', 'beneath', 'marked_comparison'])
def test_public_user_fields(mode, task):
    text = build_user_prompt(mode, question='Original question {literal}',
                             question_type=task, answer_options=['A: yes', 'B: no'],
                             step=2, max_steps=35, last_tool='perceive_find_object',
                             previous_tool_response='Result {untouched}', remaining_budget=7)
    assert 'Question: Original question {literal}' in text
    assert f'Task: {"compare" if task == "marked_comparison" else task}' in text
    assert ('Answer options:' in text) == (task != 'count')
    if mode == 'direct':
        assert 'Status:' not in text
        assert 'Previous tool response:' not in text
    else:
        assert 'Step: 2 of 35' in text
        assert 'Last tool: find_object' in text
        assert 'Previous tool response:\nResult {untouched}' in text
    assert 'Search guidance' not in text
    assert 'badge_points' not in text
    assert 'MUST physically remove' not in text


@pytest.mark.parametrize('mode', ['perception', 'agentic'])
def test_first_turn_omits_previous_result(mode):
    text = build_user_prompt(mode, question='Q', question_type='find',
                             previous_tool_response='must not appear', step=1)
    assert 'Previous tool response' not in text
    assert 'must not appear' not in text
    assert 'Last tool: none' in text


def test_ground_truth_is_not_a_template_input():
    with pytest.raises(TypeError):
        build_user_prompt('agentic', question='Q', question_type='count', gt_answer=3)
    text = format_previous_tool_response(
        {'tool': 'grasp_and_remove', 'object_name': 'SECRET_GT_NAME',
         'gt_answer': 'SECRET_GT_ANSWER', 'success': False,
         'grasp_outcome': 'grasped_slipped'})
    assert 'SECRET' not in text
    assert 'slipped' in text
    assert 'Tool: grasp' in text


def test_previous_results_distinguish_execution_from_success():
    text = format_previous_tool_response({'tool': 'push', 'success': True,
                                         'direction': 'left', 'distance_cm': 10})
    assert 'Sweep executed' in text
    assert 'not guaranteed' in text
    assert 'Selection valid: no' in text
    text = format_previous_tool_response({'tool': 'perceive_scene'},
                                         scene_graph_text='objects: []')
    assert 'objects: []' in text


def test_published_calls_map_to_internal_selection_without_mutating_raw_response():
    raw = {'tool': 'grasp', 'reasoning': 'clear'}
    assert normalize_planner_call(raw) == {
        'tool': 'grasp_and_remove', 'object_id': 1, 'reasoning': 'clear'}
    assert raw == {'tool': 'grasp', 'reasoning': 'clear'}
    assert normalize_planner_call({'tool': 'push', 'direction': 'left',
                                   'distance_cm': 19})['distance_cm'] == 10
    assert normalize_planner_call({'tool': 'find_object', 'point': [20, 30]}) == {
        'tool': 'perceive_find_object', 'point': [20, 30]}


@pytest.mark.parametrize('mode', ['direct', 'perception', 'agentic'])
def test_all_system_examples_are_valid_json_and_supported(mode):
    text = load_system_prompt(mode)
    examples = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    assert examples
    for example in examples:
        assert 'reasoning' in example
        if mode == 'direct':
            assert 'tool' not in example
            assert 'answer' in example
        else:
            assert example['tool'] in {'perceive_scene', 'find_object', 'grasp', 'push', 'answer_question'}
            if mode == 'perception':
                assert example['tool'] not in {'grasp', 'push'}
    assert '"plan"' not in text and '"badge_points"' not in text


def test_singlepass_release_path_does_not_use_hidden_task_guidance():
    from mgvqa.singlepass import build_singlepass_prompt
    with patch('mgvqa.singlepass.build_type_specific_constraint', side_effect=AssertionError):
        text = build_singlepass_prompt('Which is smallest?', 'marked_comparison',
                                      ['A', 'B', 'C'], {'target_name': 'SECRET', 'gt_answer': 'SECRET'})
    assert 'SECRET' not in text
    assert 'Task: compare' in text
    assert 'badge_points' not in text


@pytest.mark.parametrize('mode', ['agentic', 'perception'])
def test_real_loop_uses_templates_on_every_turn(tmp_path, mode):
    import copy
    import numpy as np
    from unittest.mock import MagicMock
    from mgvqa import reasoning_loop as loop

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    seg = np.zeros((8, 8), dtype=np.int32)
    demo = MagicMock()
    demo.env.obj_ids = []
    demo.env.camera.get_cam_img.return_value = (rgb, None, seg)
    demo.update.return_value = {'seg': seg}
    captured = []

    def model(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        return ({'tool': 'perceive_scene', 'reasoning': 'inspect'} if len(captured) == 1
                else {'tool': 'answer_question', 'answer': 'A', 'reasoning': 'done'})

    def backend(name, **kwargs):
        if name == 'perceive_scene':
            return ({'objects': [{'obj_id': 1, 'name': 'can', 'point_norm': [0.5, 0.5]}],
                     'scene_summary': 'visible scene'}, rgb, {}, [])
        return {'answer': kwargs['response']['answer'], 'point': None, 'ratio': None}

    with patch.object(loop, 'call_vlm', side_effect=model), \
            patch.object(loop, 'execute_backend', side_effect=backend), \
            patch.object(loop, '_capture_sim_gt_state'), \
            patch.object(loop, 'reencode_video'):
        result = loop.run_vlm_reasoning_loop(
            demo, {'question': 'Original Q', 'question_type': 'marked_comparison',
                   'gt_answer': 'SECRET', 'answer_options': ['A', 'B', 'C']},
            str(tmp_path / 'ep' / 'test'), None, None, None, verbose=False, mode=mode)
    assert result['status'] == 'answered'
    assert len(captured) == 2
    assert 'Previous tool response:' not in captured[0]['user_prompt']
    for turn in captured:
        assert 'Question: Original Q' in turn['user_prompt']
        assert 'Task: compare' in turn['user_prompt']
        assert 'SECRET' not in turn['user_prompt']
        assert 'badge_points' not in turn['user_prompt']
        assert turn['system_prompt'] == load_system_prompt(mode)
    assert 'visible scene' in captured[1]['user_prompt']
    assert len(captured[1]['extra_messages']) == 2
    assert len(captured[1]['extra_messages'][0]['images']) == 1


@pytest.mark.parametrize('mode', ['agentic', 'perception'])
@pytest.mark.parametrize('outcome', ['found', 'not_visible', 'invalid'])
def test_find_overlay_and_anonymous_result(tmp_path, mode, outcome):
    import copy
    import cv2
    import numpy as np
    from unittest.mock import MagicMock
    from mgvqa import reasoning_loop as loop

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay = np.full_like(rgb, 80)
    mask = np.ones((64, 64), dtype=bool)
    demo = MagicMock()
    demo.env.obj_ids = []
    demo.env.camera.get_cam_img.return_value = (rgb, None, mask.astype(np.int32))
    captured = []
    # Start with a successful selection, then replace it (including rejection).
    calls = [{'tool': 'find_object', 'point': [500, 500]},
             {'tool': 'find_object', **({} if outcome == 'invalid' else {'point': [400, 400]})},
             {'tool': 'answer_question', 'answer': 'A'}]
    def model(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        return calls[len(captured) - 1]
    def backend(name, **kwargs):
        if name == 'answer_question':
            return {'answer': 'A', 'point': None, 'ratio': None}
        found = len(captured) == 1 or outcome == 'found'
        return (overlay, {1: 'SECRET_BACKEND_LABEL'},
                [{'x': 32, 'y': 32, 'mask': mask, 'label': 'SECRET_BACKEND_LABEL'}], 'found') if found else (rgb, {}, [], 'not_visible')
    with patch.object(loop, 'call_vlm', side_effect=model), \
            patch.object(loop, 'execute_backend', side_effect=backend), \
            patch.object(loop, '_capture_sim_gt_state'), \
            patch.object(loop, 'reencode_video'):
        result = loop.run_vlm_reasoning_loop(
            demo, {'question': 'Q', 'question_type': 'find', 'gt_answer': 'SECRET_GT',
                   'target_name': 'SECRET_TARGET', 'answer_options': ['A: yes', 'B: no']},
            str(tmp_path / 'ep' / 'test'), None, None, None, verbose=False, mode=mode)
    assert result['status'] == 'answered'
    assert [len(c['images']) for c in captured] == [1, 2, 2 if outcome == 'found' else 1]
    for i, c in enumerate(captured):
        assert 'SECRET' not in c['user_prompt']
        np.testing.assert_array_equal(c['images'][0], rgb)
        assert ('Two images are attached' in c['user_prompt']) == (len(c['images']) == 2)
        if len(c['images']) == 2:
            assert 'Object detected.' in c['user_prompt']
            assert 'Image 2' in c['user_prompt']
            np.testing.assert_array_equal(c['images'][1], overlay)
            saved = cv2.imread(str(tmp_path / 'ep' / 'vlm_steps' / f'step_{i+1}_img2.png'))
            np.testing.assert_array_equal(saved, overlay)
    if outcome != 'found':
        assert 'No object detected' in captured[2]['user_prompt']
        assert not (tmp_path / 'ep' / 'vlm_steps' / 'step_3_img2.png').exists()
    assert len(captured[2]['extra_messages'][2]['images']) == 2
    np.testing.assert_array_equal(captured[2]['extra_messages'][2]['images'][1], overlay)


@pytest.mark.parametrize('bad_mask', [None, [], [[False]]])
def test_grounding_rejects_invalid_masks(bad_mask):
    import numpy as np
    from perception.grounding import perceive_find_object_at_point
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    with patch('perception.grounding._sam_segment', return_value=bad_mask):
        _, markers, detections, status = perceive_find_object_at_point(
            rgb, [500, 500], None, verbose=False)
    assert status == 'not_visible'
    assert not markers and not detections


def test_find_feedback_ignores_all_identity_metadata():
    text = format_previous_tool_response(
        {'tool': 'perceive_find_object', 'success': True, 'object_name': 'SECRET'},
        segmentation_text='name: SECRET', has_grounding=True)
    assert text == 'Tool: find_object\nObject detected.'


def test_editable_user_template_is_used(tmp_path):
    (tmp_path / 'direct_user.txt').write_text('CUSTOM {task}: {question}')
    with patch('mgvqa.prompts._PROMPT_DIR', str(tmp_path)):
        assert build_user_prompt('direct', question='Q', question_type='count') == 'CUSTOM count: Q'
