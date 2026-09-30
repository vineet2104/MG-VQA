# Models and provider keys

Choose any planner with `./run_mgvqa.sh --model MODEL --mode MODE` or
`python -m mgvqa --model MODEL --mode MODE`. Modes are `direct`, `perception`,
and `agentic`; only the shell wrapper accepts comma-separated mode lists.
Add `--dataset /path/to/MG-VQA-Bench --workers 4` to evaluate your entire downloaded
bank. No episode selection is required.

## Release planner catalog

| Planner | Exact model ID | Gateway API |
|---|---|---|
| Gemini 3.1 Pro | `gcp/google/gemini-3.1-pro-preview` | Chat Completions |
| Gemini 3.8 Flash | `gcp/google/gemini-3.8-flash` | Chat Completions |
| Nemotron 3.5 Super VL | `nvidia/nvidia/nemotron-3.5-super-vl-preview` | Chat Completions |
| Qwen 3.8 Flash Next | `nvidia/qwen/qwen3.8-flash-next` | Chat Completions |
| Grok 4.6 | `xai/xai/grok-4.6` | Chat Completions |
| GPT-5.6 Sol | `azure/openai/gpt-5.6-sol` | Responses |
| GPT-6 Astra | `azure/openai/gpt-6-astra` | Responses |
| Cosmos3 Nano Reasoner | `nvidia/nvidia/cosmos3-nano-reasoner` | Chat Completions |
| Claude Opus 4.8 | `aws/anthropic/bedrock-claude-opus-4-8` | Messages |
| Gemma 4 31B IT | `nvidia/google/gemma-4-31b-it` | Chat Completions |
| Qwen 3.6 35B A3B | `nvidia/qwen/qwen3.6-35b-a3b` | Chat Completions |
| Kimi K3 | `nvidia/moonshotai/kimi-k3` | Chat Completions |
| GLM 5.3 | `nvidia/zai-org/glm-5.3` | Chat Completions |

Default planner: Gemini 3.1 Pro. The separate scene-graph backend defaults to
`gcp/google/gemini-3.6-flash` (`--scene-model`); SAM3 handles point segmentation.
All 13 planners and the scene model passed tiny image probes on 2026-09-26.
Real-scene, long-history reliability is still pending; account access may vary.

```bash
# Compare planners under identical modes, questions and scene backend
for model in gcp/google/gemini-3.1-pro-preview azure/openai/gpt-6-astra; do
  ./run_mgvqa.sh --model "$model" --mode direct,perception,agentic \
    --dataset /path/to/MG-VQA-Bench --workers 4
done
```

## NVIDIA gateway (release default)

Set `NV_API_KEY` or `NVIDIA_API_KEY`. Full gateway model IDs are passed unchanged.
`models/catalog.py` selects Chat Completions, Responses, or Anthropic Messages.
`azure/`, `gcp/`, `aws/`, `nvidia/`, `openai/` and `xai/` route to this gateway.
In particular, **`openai/` is a gateway prefix, not a direct OpenAI selector**.

```bash
./run_mgvqa.sh --model azure/openai/gpt-6-astra --mode agentic
python scripts/check_models.py --all  # small paid image+history probes
```

Protocol adapters preserve prior text/image turns. Planner system prompts do
not change by model. Provider-specific transport options (such as unsupported
temperature fields) are separate from the prompts. Availability and image
support must be validated on your account; a catalog entry alone is insufficient.

## Other API keys

```bash
# Direct Google: bare gemini-* model name
# Set GEMINI_API_KEY (or GOOGLE_API_KEY)
./run_mgvqa.sh --model gemini-3.1-pro-preview --mode direct

# Direct OpenAI: bare model name (Chat Completions adapter)
# Set OPENAI_API_KEY
./run_mgvqa.sh --model gpt-4o --mode direct

# Direct Anthropic: anthropic/<native Anthropic model ID>
# Set ANTHROPIC_API_KEY; choose an ID available to your account
./run_mgvqa.sh --model anthropic/claude-opus-4-8 --mode direct
```

Native IDs and release availability may differ from gateway IDs. The direct
OpenAI adapter currently targets Chat Completions; Responses-only models should
use the supported NVIDIA Responses route until a native adapter is added.
Perception/agentic runs still use the fixed NVIDIA scene model by default. To
avoid NVIDIA entirely, set `--scene-model gemini-3.6-flash` and provide a Google
key (with access to that native model). All compared planners should use the
same perception backend and prompts.

Existing optional routes: `grok-*` → XAI_API_KEY; `qwen/`, `google/gemma-`,
`deepseek/` → OPENROUTER_API_KEY. These are not the release model catalog.

## Custom provider

For a custom Chat Completions server, set `CUSTOM_BASE_URL` (ending in `/v1`)
and `CUSTOM_API_KEY`, then use a model name without a reserved prefix.

To add a new protocol/provider, implement `models/providers/<provider>.py`
and register a prefix in `models/dispatch.py`. The interface is:

```python
def request(images, prompt, system_prompt, model_name, temp=0.0,
            n_tokens=8192, disable_thinking=False, extra_messages=None):
    # images: RGB uint8 arrays
    # history: [{"role": "user"|"assistant", "text": str, "images": [array]}]
    # Return final response text, not provider metadata.
    ...
```

Do not drop history images, silently substitute a model, or add model-specific
system-prompt instructions. Test request serialization, final-text extraction,
rate limits, malformed output and an actual multimodal request.
