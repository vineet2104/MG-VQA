"""
models.singlepass_augment — Model-specific system prompt augmentation for singlepass mode.
"""

GEMMA_SINGLEPASS_REINFORCEMENT = """

== CRITICAL: OUTPUT FORMAT REMINDER ==
You MUST respond with EXACTLY ONE JSON object and NOTHING else.
NO prose. NO markdown. NO code fences. NO explanations outside the JSON.
Even when uncertain, ALWAYS return valid JSON — never refuse, never apologise.

Concrete examples of the only acceptable response shapes:
- counting / interactive_counting / under_object:
  {"answer": "C", "reasoning": "I count 3 red bowls in the scene."}
- existence (when answer is "yes", include a point):
  {"answer": "A", "point": [340, 520], "reasoning": "Turquoise bowl visible at center-right."}
- existence (when answer is "no"):
  {"answer": "B", "reasoning": "No turquoise bowl is visible anywhere on the table."}
- same_attribute:
  {"answer": "A", "reasoning": "Both A and B appear red."}
- relative_size (always include a ratio):
  {"answer": "A", "ratio": 2.5, "reasoning": "A is roughly 2.5x larger than B."}

If A/B coordinates appear blank like '[ ,  ]', the object is occluded
in this view — make your best guess from what you can see; do NOT
refuse to answer.
""".rstrip() + "\n"


def augment_singlepass_system_prompt(system_prompt: str, model_name: str) -> str:
    """Return the singlepass system_prompt with model-specific additions.

    Currently only Gemma 4 receives an addition — its weaker JSON-formatting
    compliance benefits from explicit examples and a "never refuse" instruction.
    Every other model is returned unchanged.
    """
    m = (model_name or "").lower()
    if m.startswith("google/gemma-"):
        return system_prompt + GEMMA_SINGLEPASS_REINFORCEMENT
    return system_prompt
