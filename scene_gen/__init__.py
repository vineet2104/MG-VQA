"""Scene generation package for Agentic MQA.

Runtime module:
    prompt_variants — 120 prompt-template variants across the 6 question
                      types. One random non-canonical variant is applied
                      per episode (see questions.py :: _apply_random_variant).

The offline scene-generation toolkit (catalog, sampler, poser, validator,
io, etc.) was archived to scripts-to-ignore/ — it is not part of the
benchmark runtime.
"""
