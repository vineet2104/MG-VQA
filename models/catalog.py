"""Release model identifiers and NVIDIA gateway wire protocols.

Catalog presence is not a guarantee of account access or multimodal support.
Use scripts/check_models.py to test the actual image + conversation interface.
"""

NVIDIA_MODELS = {
    "gcp/google/gemini-3.1-pro-preview": ("Gemini 3.1 Pro", "chat"),
    "gcp/google/gemini-3.8-flash": ("Gemini 3.8 Flash", "chat"),
    "nvidia/nvidia/nemotron-3.5-super-vl-preview": ("Nemotron 3.5 Super VL", "chat"),
    "nvidia/qwen/qwen3.8-flash-next": ("Qwen 3.8 Flash Next", "chat"),
    "xai/xai/grok-4.6": ("Grok 4.6", "chat"),
    "azure/openai/gpt-5.6-sol": ("GPT-5.6 Sol", "responses"),
    "azure/openai/gpt-6-astra": ("GPT-6 Astra", "responses"),
    "nvidia/nvidia/cosmos3-nano-reasoner": ("Cosmos3 Nano Reasoner", "chat"),
    "aws/anthropic/bedrock-claude-opus-4-8": ("Claude Opus 4.8", "messages"),
    "nvidia/google/gemma-4-31b-it": ("Gemma 4 31B IT", "chat"),
    "nvidia/qwen/qwen3.6-35b-a3b": ("Qwen 3.6 35B A3B", "chat"),
    "nvidia/moonshotai/kimi-k3": ("Kimi K3", "chat"),
    "nvidia/zai-org/glm-5.3": ("GLM 5.3", "chat"),
    "gcp/google/gemini-3.6-flash": ("Gemini 3.6 Flash (perception)", "chat"),
}
DEFAULT_PLANNER = "gcp/google/gemini-3.1-pro-preview"
DEFAULT_SCENE_MODEL = "gcp/google/gemini-3.6-flash"


def gateway_protocol(model_name):
    return NVIDIA_MODELS.get(model_name, (model_name, "chat"))[1]
