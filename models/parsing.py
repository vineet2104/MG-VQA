"""
models.parsing — VLM JSON response parsing with truncation repair.
"""

import re
import json


def parse_vlm_response(response_text: str) -> dict:
    """Parse the VLM's JSON response, extracting the **first** valid JSON object.

    Handles:
      - Markdown code fences (```json ... ```)
      - Multiple JSON objects on separate lines (takes the first)
      - Nested braces within a single object
      - Truncated JSON (best-effort repair)

    Returns a dict with at least {"tool": "..."}, or {"tool": "error"} on failure.
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try parsing the full text first (ideal case: single JSON object)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Extract the first balanced {...} from the text.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    # Truncation-repair fallback
    if start != -1:
        snippet = text[start:]
        repaired = _repair_truncated_json(snippet)
        if repaired is not None:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    parsed["_truncated_and_repaired"] = True
                    return parsed
            except json.JSONDecodeError:
                pass

    return {"tool": "error", "raw_response": response_text}


def _repair_truncated_json(snippet):
    """Best-effort repair for a JSON object truncated mid-stream.

    Closes any dangling string, drops a trailing partial key/value, and
    appends the right number of ]/} to balance brackets.
    Returns None if the snippet can't be a JSON object start.
    """
    if not snippet or snippet[0] != "{":
        return None

    in_string = False
    escape = False
    stack = []
    last_comma = -1

    i = 0
    while i < len(snippet):
        ch = snippet[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                if not stack:
                    return snippet[: i + 1]
            elif ch == "," and len(stack) == 1:
                last_comma = i
        i += 1

    repaired = snippet
    if in_string:
        repaired = repaired + '"'

    if last_comma > 0:
        tail = repaired[last_comma + 1:]
        if ":" not in tail or tail.count('"') % 2 != 0:
            repaired = repaired[:last_comma]

    closers = {"{": "}", "[": "]"}
    for opener in reversed(stack):
        repaired += closers[opener]

    return repaired
