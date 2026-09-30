"""Wire-format adapters; preserve every text/image turn, independent of prompts."""
from models.utils import encode_image_to_base64


def messages_for(protocol, images, prompt, extra_messages=None):
    history = list(extra_messages or []) + [
        {"role": "user", "text": prompt, "images": images}]
    result = []
    for message in history:
        role = message["role"]
        parts = []
        for image in message.get("images") or []:
            b64 = encode_image_to_base64(image)
            if protocol == "messages":
                parts.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}})
            elif protocol == "responses":
                parts.append({"type": "input_image", "image_url":
                              f"data:image/jpeg;base64,{b64}"})
            else:
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}})
        text = message.get("text", "")
        if text:
            kind = ("output_text" if role == "assistant" else "input_text") \
                if protocol == "responses" else "text"
            parts.append({"type": kind, "text": text})
        result.append({"role": role, "content": parts})
    return result


def build_payload(protocol, model, images, prompt, system_prompt,
                  n_tokens=8192, extra_messages=None):
    messages = messages_for(protocol, images, prompt, extra_messages)
    if protocol == "responses":
        return {"model": model, "instructions": system_prompt, "input": messages,
                "max_output_tokens": n_tokens, "store": False}
    if protocol == "messages":
        return {"model": model, "system": system_prompt, "messages": messages,
                "max_tokens": n_tokens}
    return {"model": model, "messages": [
        {"role": "system", "content": system_prompt}, *messages],
        "max_tokens": n_tokens}


def response_text(protocol, data):
    if protocol == "responses":
        return "\n".join(part["text"] for item in data.get("output", [])
                         if item.get("type") == "message"
                         for part in item.get("content", [])
                         if part.get("type") == "output_text")
    if protocol == "messages":
        return "\n".join(part["text"] for part in data.get("content", [])
                         if part.get("type") == "text")
    msg = data["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or ""
    if isinstance(text, list):
        text = "\n".join(p.get("text", "") for p in text if p.get("type") == "text")
    return text
