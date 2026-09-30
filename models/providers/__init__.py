"""
models.providers — Backend implementations for each VLM API provider.

Each provider module exposes a single `request()` function with the signature:

    def request(
        images: List[np.ndarray],
        prompt: str,
        system_prompt: str,
        model_name: str,
        temp: float = 0.0,
        n_tokens: int = 8192,
        disable_thinking: bool = False,
        extra_messages: List[dict] = None,
    ) -> str:
        ...
"""
