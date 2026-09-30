"""Public tool contracts and lazy backend dispatch.

The reasoning loop owns budgets, marker freshness, history and termination.
Backends own perception/physics; they never receive the question's GT answer.
Add a ToolSpec here and a lifecycle handler in reasoning_loop.py for a new tool.
"""
from dataclasses import dataclass
from importlib import import_module
from typing import Tuple


@dataclass(frozen=True)
class ToolSpec:
    name: str
    backend: str
    modes: Tuple[str, ...]
    required_fields: Tuple[str, ...] = ()
    manipulation: bool = False
    terminates: bool = False


TOOLS = {
    t.name: t for t in (
        ToolSpec('perceive_scene', 'perception.scene_graph:perceive_scene_v2',
                 ('perception', 'agentic')),
        ToolSpec('perceive_find_object', 'perception.grounding:perceive_find_object_at_point',
                 ('perception', 'agentic'), ('point',)),
        ToolSpec('grasp_and_remove', 'manipulation.grasp:execute_grasp_and_remove',
                 ('agentic',), manipulation=True),
        ToolSpec('push', 'manipulation.push:execute_push',
                 ('agentic',), ('direction',), manipulation=True),
        ToolSpec('answer_question', 'mgvqa.answer:answer_question',
                 ('direct', 'perception', 'agentic'), ('answer',), terminates=True),
    )
}
ALIASES = {'find_object': 'perceive_find_object', 'answer_questions': 'answer_question',
           'grasp': 'grasp_and_remove'}


def public_name(name):
    return {'perceive_find_object': 'find_object', 'grasp_and_remove': 'grasp'}.get(name, name)


def normalize_planner_call(response):
    """Map published argument-free selection to the internal marker contract.

    Only marker 1 is implicitly selected; existing loop checks still reject a
    missing/stale segmentation. The published push is always 10 cm. Physics
    backends retain their configurable API for non-planner callers.
    """
    result = dict(response)
    result['tool'] = canonical_name(result.get('tool', 'error'))
    if result['tool'] in {'grasp_and_remove', 'push'}:
        result.setdefault('object_id', 1)
    if result['tool'] == 'push':
        result['distance_cm'] = 10
    return result


def canonical_name(name):
    return ALIASES.get(name, name)


def available_tools(mode):
    mode = 'direct' if mode == 'singlepass' else mode
    if mode not in {'direct', 'perception', 'agentic'}:
        raise ValueError(f'Unknown mode: {mode}')
    return tuple(t for t in TOOLS.values() if mode in t.modes)


def execute_backend(name, mode='agentic', **kwargs):
    """Invoke a backend using internal arguments, after loop-side validation."""
    spec = TOOLS[canonical_name(name)]
    if spec not in available_tools(mode):
        raise ValueError(f'{spec.name} is unavailable in {mode} mode')
    module, symbol = spec.backend.split(':')
    return getattr(import_module(module), symbol)(**kwargs)
