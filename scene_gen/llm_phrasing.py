"""LLM-based question phrasing for the V2 benchmark (count / find / beneath).

The deterministic generators in ``questions.py`` decide the ground truth
(target object, count, cover/hidden pair, options).  This module optionally
replaces the *surface phrasing* of the question with a natural, human-sounding
sentence produced by an LLM — while the machine-readable answer-format
instruction (the "type-specific suffix") is appended deterministically so
parsing/scoring never depend on the LLM.

Design guarantees
-----------------
* The LLM only rewrites the question STEM.  Ground truth (``gt_answer``,
  ``answer_options``, ``option_map``, ``target_obj_id`` …) is never touched.
* The answer-format suffix (point grid for ``find``, "single integer" for
  ``count``, the "none of the above" note for ``beneath``) is fixed text —
  the LLM must not paraphrase it.
* Deterministic + cheap: results are cached on disk keyed by
  (qtype, slots) so a given scene bank is reproducible and we do not hit the
  API once per episode on every run.  A cache hit costs nothing.
* Robust: any API failure falls back to a clean template stem, so the
  pipeline never crashes because the LLM was unavailable.

Only used when ``--question_phrasing llm`` is set (default ``template`` keeps
the existing behaviour byte-identical).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Type-specific suffixes — DETERMINISTIC.  Appended verbatim after the LLM
# stem.  These carry the answer-format contract the parser/scorer rely on and
# must match what the template pipeline produced.
# ---------------------------------------------------------------------------
FIND_SUFFIX = (
    " If yes, also point to it by providing the (row, col) coordinates "
    "normalised to a [0, 1000] grid."
)
# count / beneath stems are self-contained — the answer-format contract for
# those is carried by build_type_specific_constraint()/answer_options, so no
# extra text is appended to the stem itself (mirrors the template pipeline).
TYPE_SUFFIX = {
    "find": FIND_SUFFIX,
    "count": "",
    "beneath": "",
    # compare (marked_comparison): the A/B/C answer-format contract is carried
    # by build_type_specific_constraint()/answer_options, so nothing is
    # appended to the stem.  The stem must still name the badges A, B, C.
    "compare": "",
}

_SYSTEM_PROMPT = (
    "You write short, natural questions that a person standing at a cluttered "
    "tabletop would casually ask another person. Vary the phrasing so it does "
    "not sound templated. Never invent facts, never add answer options, never "
    "add instructions about how to answer or point. Output ONLY the question "
    "sentence — no preamble, no quotes, no explanation."
)


def _describe_object(name: str, details: Optional[dict]) -> str:
    """Compact natural description of the object for the meta-prompt."""
    if not details:
        return name
    color = (details.get("color") or "").strip()
    category = (details.get("shape") or details.get("category") or "").strip()
    bits = [f"display name: {name!r}"]
    if color:
        bits.append(f"color: {color}")
    if category and category not in name:
        bits.append(f"category: {category}")
    return "; ".join(bits)


def _meta_prompt(qtype: str, slots: dict) -> str:
    """Build the per-type instruction sent to the LLM."""
    if qtype == "count":
        obj = _describe_object(slots.get("item", ""), slots.get("item_details"))
        return (
            "Task type: COUNTING.\n"
            f"Object to count ({obj}).\n"
            "Write one natural question asking how many objects that look "
            "like this object are on the table. Refer to the object by its "
            "display name naturally (you may lightly rephrase, e.g. 'green "
            "measuring cup' → 'green measuring cups'), but keep the same "
            "object. Do NOT mention a number. Do NOT say 'resemble'."
        )
    if qtype == "find":
        obj = _describe_object(slots.get("item", ""), slots.get("item_details"))
        return (
            "Task type: PRESENCE CHECK.\n"
            f"Object to look for ({obj}).\n"
            "Write one natural yes/no question asking whether this object is "
            "on the table. Refer to it by its display name naturally. Do NOT "
            "add any instruction about pointing or coordinates."
        )
    if qtype == "beneath":
        cover = _describe_object(slots.get("cover", ""),
                                 slots.get("cover_details"))
        return (
            "Task type: OCCLUSION / WHAT-IS-UNDERNEATH.\n"
            f"Covering object ({cover}).\n"
            "Write one natural question asking which object is located "
            "directly underneath the covering object. Refer to the covering "
            "object by its display name naturally. Do NOT list options."
        )
    if qtype == "compare":
        _ext = slots.get("extremum", "largest")
        _word = "SMALLEST" if _ext == "smallest" else "LARGEST"
        return (
            "Task type: MARKED SIZE COMPARISON.\n"
            "Three objects each carry a small painted badge: a circle labelled "
            "A, a square labelled B, and a triangle labelled C.\n"
            f"Write one natural question asking which of the three badged "
            f"objects (A, B, or C) is the {_word} by physical volume. You MUST "
            "keep the three labels A, B, and C in the question so the answer "
            "options still make sense. Do NOT reveal or hint at the answer, "
            "do NOT describe the objects, do NOT list options beyond the "
            "labels A/B/C."
        )
    return ""


# ---------------------------------------------------------------------------
# Deterministic template fallback (also used when --question_phrasing template
# is NOT active but the LLM call fails).
# ---------------------------------------------------------------------------
def _template_stem(qtype: str, slots: dict) -> str:
    if qtype == "count":
        return (f"How many objects that resemble a {slots.get('item','')} "
                f"do you see on the table?")
    if qtype == "find":
        return f"Is there a {slots.get('item','')} on the table?"
    if qtype == "beneath":
        return f"What object is under the {slots.get('cover','')}?"
    if qtype == "compare":
        if slots.get("extremum") == "smallest":
            return "Which labelled object has the smallest volume: A, B, or C?"
        return "Which labelled object has the largest volume: A, B, or C?"
    return slots.get("item", "")


# ---------------------------------------------------------------------------
# Disk cache — keyed by (qtype, normalized slots).  Thread/process safe enough
# for our use (worst case a duplicate API call; last write wins).
# ---------------------------------------------------------------------------
_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache", "llm_question_phrasings.json")
_mem_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()
_cache_loaded = False


def _cache_key(qtype: str, slots: dict) -> str:
    payload = {
        "qtype": qtype,
        "item": slots.get("item", ""),
        "cover": slots.get("cover", ""),
        # compare polarity — largest vs smallest must NOT share a cache entry.
        "extremum": slots.get("extremum", ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load_cache(path: str) -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    with _cache_lock:
        if _cache_loaded:
            return
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    _mem_cache.update(json.load(f))
        except Exception:
            pass
        _cache_loaded = True


def _save_cache(path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _cache_lock:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_mem_cache, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
    except Exception:
        pass


def _sanitize(text: str) -> str:
    """Strip quotes / preamble the LLM might add; keep one line."""
    if not text:
        return ""
    t = text.strip()
    # take the first non-empty line
    for line in t.splitlines():
        line = line.strip()
        if line:
            t = line
            break
    # drop wrapping quotes
    t = t.strip().strip('"').strip("'").strip()
    # drop a leading "Question:" style prefix
    t = re.sub(r"^(question|q)\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def _looks_valid(stem: str, qtype: str, slots: dict) -> bool:
    """Cheap sanity checks so we never save a broken/echoed prompt."""
    if not stem or len(stem) < 6 or len(stem) > 240:
        return False
    if "\n" in stem:
        return False
    low = stem.lower()
    # must be a question
    if "?" not in stem:
        return False
    # must not have leaked answer-format instructions
    banned = ("row, col", "[0, 1000]", "grid", "coordinate", "option",
              "none of the above", "answer with", "single integer")
    if any(b in low for b in banned):
        return False
    # compare: the three badge labels A, B, C must all survive, otherwise the
    # A/B/C answer options become meaningless.
    if qtype == "compare":
        import re as _re
        for _lab in ("A", "B", "C"):
            if not _re.search(rf"\b{_lab}\b", stem):
                return False
        # GT-SAFETY: the stem's size polarity MUST match the requested extremum.
        # A mis-phrased direction would silently invert the ground truth, so we
        # reject it here and fall back to the exact template.  We require a
        # correct-direction keyword AND the absence of any opposite-direction
        # keyword.
        _ext = slots.get("extremum", "largest")
        _LARGE = ("largest", "biggest", "greatest", "largest-volume",
                  "most voluminous", "bulkiest", "most volume", "greatest volume",
                  "largest volume", "most space", "greatest size", "largest size")
        _SMALL = ("smallest", "tiniest", "least", "smallest-volume",
                  "least voluminous", "littlest", "smallest volume",
                  "least volume", "least space", "smallest size", "least size")
        want, avoid = ((_SMALL, _LARGE) if _ext == "smallest"
                       else (_LARGE, _SMALL))
        if not any(w in low for w in want):
            return False
        if any(a in low for a in avoid):
            return False
    return True


def phrase_question(
    qtype: str,
    slots: dict,
    model_name: str,
    *,
    cache_path: Optional[str] = None,
    temperature: float = 0.7,
    use_llm: bool = True,
) -> dict:
    """Return {'stem': str, 'question': str, 'source': 'llm'|'template'|'cache'}.

    ``slots`` provides the surface fillers plus optional ``*_details`` dicts
    (color/category/size) so the LLM can phrase naturally:
        count/find : {'item', 'item_details'}
        beneath    : {'cover', 'cover_details'}

    ``question`` = stem + deterministic type suffix.
    """
    suffix = TYPE_SUFFIX.get(qtype, "")

    def _finalize(stem: str, source: str) -> dict:
        stem = stem.strip()
        q = stem + suffix
        return {"stem": stem, "question": q, "source": source}

    if not use_llm:
        return _finalize(_template_stem(qtype, slots), "template")

    path = cache_path or _DEFAULT_CACHE
    _load_cache(path)
    key = _cache_key(qtype, slots)
    if key in _mem_cache:
        return _finalize(_mem_cache[key], "cache")

    stem = ""
    try:
        from models import call_model
        # Budget must be generous: thinking models (Gemini 3.x, Qwen3) spend
        # much of the token budget on internal reasoning, so a small cap
        # truncates the actual answer to a fragment.
        raw = call_model(
            images=[],
            prompt=_meta_prompt(qtype, slots),
            system_prompt=_SYSTEM_PROMPT,
            model_name=model_name,
            temp=temperature,
            n_tokens=1024,
            disable_thinking=True,
        )
        stem = _sanitize(raw)
    except Exception as e:      # noqa: BLE001 — never crash the pipeline
        print(f"[llm_phrasing] WARN {qtype}: {e} — using template")
        stem = ""

    if not _looks_valid(stem, qtype, slots):
        if stem:
            print(f"[llm_phrasing] rejected {qtype} phrasing {stem!r} "
                  f"— using template")
        return _finalize(_template_stem(qtype, slots), "template")

    with _cache_lock:
        _mem_cache[key] = stem
    _save_cache(path)
    return _finalize(stem, "llm")
