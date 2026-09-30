"""Fast release-contract tests; no simulator, network or GPU needed."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from models.catalog import NVIDIA_MODELS, gateway_protocol
from models.dispatch import _get_provider
from models.protocols import build_payload, response_text
from mgvqa.prompts import load_system_prompt
from mgvqa.tool_registry import available_tools, execute_backend, canonical_name


@pytest.mark.parametrize('protocol', ['chat', 'responses', 'messages'])
def test_history_and_images(protocol):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay = np.full_like(image, 200)
    payload = build_payload(protocol, 'test', [image, overlay], 'current', 'system', 123,
                            [{'role': 'user', 'text': 'old', 'images': [image, overlay]},
                             {'role': 'assistant', 'text': 'reply'}])
    turns = payload.get('input', payload.get('messages'))
    if protocol == 'chat':
        turns = turns[1:]
    assert [t['role'] for t in turns] == ['user', 'assistant', 'user']
    assert len(turns[0]['content']) == 3
    assert len(turns[-1]['content']) == 3
    assert turns[0]['content'][:2] == turns[-1]['content'][:2]
    assert turns[-1]['content'][0] != turns[-1]['content'][1]
    if protocol == 'responses':
        assert turns[1]['content'][0]['type'] == 'output_text'
        assert payload['max_output_tokens'] == 123
    else:
        assert payload['max_tokens'] == 123


def test_native_text_extraction():
    assert response_text('responses', {'output': [
        {'type': 'reasoning', 'summary': []},
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'answer'}]}]}) == 'answer'
    assert response_text('messages', {'content': [
        {'type': 'thinking', 'thinking': 'hidden'},
        {'type': 'text', 'text': 'answer'}]}) == 'answer'


@pytest.mark.parametrize('model', NVIDIA_MODELS)
def test_release_models_route_to_gateway(model):
    assert _get_provider(model).__module__ == 'models.providers.nvidia'


def test_tool_modes_and_answer():
    assert len(available_tools('agentic')) == 5
    assert len(available_tools('perception')) == 3
    assert len(available_tools('direct')) == 1
    assert canonical_name('find_object') == 'perceive_find_object'
    assert execute_backend('answer_questions', response={'answer': 0})['answer'] == 0
    with pytest.raises(ValueError):
        execute_backend('push', mode='perception')


def test_prompts_are_mode_files():
    for mode in ('direct', 'perception', 'agentic'):
        assert load_system_prompt(mode) == Path(f'mgvqa/prompt_templates/{mode}.txt').read_text()
    assert '3. grasp_and_remove(object_id)' not in load_system_prompt('perception')
    assert load_system_prompt('singlepass') == load_system_prompt('direct')


def test_scene_backend_reaches_subprocess():
    from mgvqa.orchestrator.parallel import parse_args, build_worker_cmd
    with patch.object(sys, 'argv', ['test', '--n_episodes', '600', '--episode_subset', '',
                                   '--scene_model', 'custom/scene']):
        args = parse_args()
    cmd = build_worker_cmd(args, 0, [0], '/tmp/worker', '/tmp/pause')
    assert cmd[0] == sys.executable
    assert cmd[cmd.index('--scene_model') + 1] == 'custom/scene'


def test_blender_explicit_path_is_not_silently_ignored():
    from robo_blender.server_launcher import _find_blender
    with patch.dict(os.environ, {'BLENDER': '/nonexistent/blender'}):
        with pytest.raises(FileNotFoundError):
            _find_blender()


@pytest.mark.parametrize('model,endpoint,protocol', [
    ('azure/openai/gpt-6-astra', '/responses', 'responses'),
    ('aws/anthropic/bedrock-claude-opus-4-8', '/messages', 'messages'),
    ('xai/xai/grok-4.6', '/chat/completions', 'chat'),
])
def test_gateway_transport(model, endpoint, protocol):
    from models.providers.nvidia import request
    from unittest.mock import Mock
    result = {'responses': {'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': 'ok'}]}]},
        'messages': {'content': [{'type': 'text', 'text': 'ok'}]},
        'chat': {'choices': [{'message': {'content': 'ok'}}]}}[protocol]
    response = Mock(status_code=200)
    response.json.return_value = result
    with patch.dict(os.environ, {'NV_API_KEY': '', 'NVIDIA_API_KEY': 'test'}), \
            patch('requests.post', return_value=response) as post:
        assert request([], 'question', 'system', model) == 'ok'
        assert post.call_args.args[0].endswith(endpoint)
        payload = post.call_args.kwargs['json']
        assert payload['model'] == model
        if protocol != 'chat':
            assert 'temperature' not in payload
