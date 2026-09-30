"""Three-way marked relative-size question.

Objects are identified only by texture-baked markers: A/circle, B/square and
C/triangle.  Badge locations are useful evidence and are requested as nullable
diagnostics, never as a precondition for accepting a final answer.
"""

BADGE_DESCRIPTION = (
    "Exactly three objects in the scene carry small painted badges: a white "
    "CIRCLE containing A, a white SQUARE containing B, and a white TRIANGLE "
    "containing C. The badges are printed on the objects themselves, so a "
    "badge may be hidden when that part of its object is covered."
)

QUESTION_TEMPLATE = "Which labelled object has the largest volume: A, B, or C?"
QUESTION_TEMPLATE_SMALLEST = (
    "Which labelled object has the smallest volume: A, B, or C?")


def question_for_extremum(extremum: str) -> str:
    """Return the fixed A/B/C question stem for a given comparison polarity."""
    return (QUESTION_TEMPLATE_SMALLEST if extremum == "smallest"
            else QUESTION_TEMPLATE)

# Used by the full agentic condition. This deliberately permits, rather than
# mandates, evidence gathering and contains no privileged target-protection cue.
AGENTIC_CONSTRAINT = (
    "Three objects are identified by painted badges (circle=A, square=B, "
    "triangle=C). You may inspect the scene with perception and manipulation "
    "tools. Before deciding, it may help to locate the badges and keep track of "
    "the labelled objects; if a label is obscured, manipulation may reveal more "
    "evidence. You may answer whenever you have enough evidence."
)

# Candidate *future intervention* prompts. They are intentionally not wired into
# any completed baseline arm. Both are reasoning aids, not hidden badge-location
# signals or evaluator gates: the agent retains the ability to submit an answer.
#
# Do not claim that their rendered apparent sizes are literally identical: the
# physical badge paint can be clipped by an object's usable surface. The factual
# invariant relevant to the task is that badge appearance is an identity code,
# never evidence about its host object's volume.
# Shared CORE of the "instructed" compare guidance (mode-independent): all three
# badges exist, badge graphics are a fixed-size identity code (not a volume cue),
# work systematically and compare physical sizes.  What follows the core differs
# by tool-access mode — specifically, the guidance for a STILL-HIDDEN badge:
#   • agentic     : the agent can physically reveal it, so it is told not to
#                   dismiss a hidden badge as small (it can go verify).
#   • perception  : cannot act, so it must NOT speculate that a hidden badge
#     / direct      marks the extremum; answer from visible evidence and treat
#                   an unlocatable badge as genuinely unknown.
# The core + tail are factored out so each mode appends its own middle text.
_INSTRUCTED_CORE = (
    "ALL THREE labelled badges A, B, and C are in this scene; none is absent. "
    "All badge graphics use the same standardized size and design, so a badge is "
    "an identity code—not evidence about its host object's volume. Please carefully "
    "look and answer based on what you find. Work systematically: keep track of the "
    "object carrying each A/B/C badge, identify all three whenever feasible, then "
    "compare their physical sizes and provide your answer."
)

# Agentic-only clause: safe to tell the agent not to dismiss a hidden badge,
# because it can reveal it by manipulation.  _INSTRUCTED_BODY below (core + this
# clause) is byte-identical to the original shared body used by agentic.
_HIDDEN_CLAUSE_AGENTIC = (
    " A badge may be covered or "
    "face away; do not infer that its object is absent or small when you cannot "
    "currently see it."
)
_INSTRUCTED_BODY = _INSTRUCTED_CORE + _HIDDEN_CLAUSE_AGENTIC

# No-action clause for perception / direct: the model CANNOT reveal a hidden
# badge, so it must decide from what it can actually see and must NOT assume the
# unseen badge marks the largest (or smallest) object.  This removes the earlier
# bias where "do not assume it is small" was read as "the hidden badge is large."
_HIDDEN_CLAUSE_NO_ACTION = (
    " A badge may be covered or facing away, and you cannot reveal it. Decide "
    "from the badges you can actually see and treat any badge you cannot locate "
    "as genuinely unknown: do NOT assume a hidden badge marks the largest object, "
    "nor the smallest. If you cannot identify all three, give your best judgment "
    "from the visible evidence rather than speculating about the unseen badge."
)

_INSTRUCTED_TAIL = (
    " This is a strong evidence-gathering "
    "recommendation, not an evaluator-imposed answer gate."
)

AGENTIC_INSTRUCTED_CONSTRAINT = (
    _INSTRUCTED_BODY
    + " If a badge is still hidden, use visible occlusion relations "
      "to choose a plausible cover to move."
    + _INSTRUCTED_TAIL
)

# Perception-only variant: no manipulation, so it uses the neutral no-action
# clause and does not speculate about the hidden badge's host size.
PERCEPTION_INSTRUCTED_CONSTRAINT = (
    _INSTRUCTED_CORE
    + _HIDDEN_CLAUSE_NO_ACTION
    + _INSTRUCTED_TAIL
)

# Direct / single-image variant: no tools at all — same neutral no-action clause.
SINGLEPASS_INSTRUCTED_CONSTRAINT = (
    _INSTRUCTED_CORE
    + _HIDDEN_CLAUSE_NO_ACTION
    + _INSTRUCTED_TAIL
)

AGENTIC_LEDGER_CONSTRAINT = (
    AGENTIC_INSTRUCTED_CONSTRAINT + " Keep a compact A/B/C evidence ledger in "
    "your belief: for each label, record the object identity, badge seen/unseen, "
    "and volume evidence. After several moves without a newly identified badge, "
    "re-perceive and state the next causal occluder before moving it."
)

PERCEPTION_CONSTRAINT = (
    "Three objects are identified by painted badges (circle=A, square=B, "
    "triangle=C). You may inspect the scene with perception tools but cannot "
    "change it. You may answer whenever you have enough evidence."
)

BADGE_SEARCH_SUGGESTION = (
    "Before deciding, it may help to carefully locate the painted A, B, and C "
    "badges so you can identify the three objects."
)

BADGE_REPORT_REQUEST = (
    "In your final answer, also include badge_points for any badges you can "
    "locate. Use null for an unseen badge; this is diagnostic and does not "
    "prevent an answer."
)

SINGLEPASS_CONSTRAINT = (
    "Three objects carry painted badges: a white CIRCLE containing A, a white "
    "SQUARE containing B, and a white TRIANGLE containing C. You have no tools "
    "and cannot move anything, so judge from this single image."
)

# Retained as an optional legacy control (--sp_identify_stage 1). It is not a
# gate: stage 2 still receives and answers from the same unchanged image.
IDENTIFY_STAGE1_PROMPT = (
    "Three objects in this image carry painted badges: a white CIRCLE containing "
    "A, a white SQUARE containing B, and a white TRIANGLE containing C. Badges "
    "are small. Search carefully and report where you can read each one.\n"
    "Respond with ONLY JSON: {\"badge_points\": {\"A\": [row, col] or null, "
    "\"B\": [row, col] or null, \"C\": [row, col] or null}, \"notes\": "
    "\"<what you found or why a badge is hidden>\"}.\n"
    "Coordinates are normalised to [0,1000] (row=0 top, col=0 left). Use null "
    "when you genuinely cannot read a badge; do not guess."
)


def identify_stage2_addendum(report: dict) -> str:
    """Feed the model's optional first-pass badge report into stage two."""
    import json as _json
    return ("== YOUR BADGE REPORT (previous step) ==\n"
            + _json.dumps(report, ensure_ascii=False)
            + "\nUse this as evidence to identify A, B, and C. Null means the "
            "badge was not readable in the unchanged image; still give your "
            "best answer.")


SINGLEPASS_SYSTEM_PROMPT = (
    "You are a careful visual reasoning assistant looking at a single "
    "top-down photograph of a cluttered tabletop. Answer only from what is "
    "visible in the image. Reply with STRICT JSON and nothing else."
)


def build_question(triple, gt: dict) -> dict:
    """Build the descriptor for three selected, already-labelled sites."""
    by_label = {s["label"]: s for s in triple}
    labels = ("A", "B", "C")
    extremum = gt.get("comparison_extremum", "largest")
    return {
        "question_type": "marked_comparison",
        "question_subtype": ("size_three_way_smallest"
                             if extremum == "smallest" else "size_three_way"),
        "question": question_for_extremum(extremum),
        "comparison_extremum": extremum,
        "badge_description": BADGE_DESCRIPTION,
        "answer_options": list(labels),
        "gt_answer": gt["gt_letter"],
        "gt_volumes_cm3": {L: by_label[L]["volume_cm3"] for L in labels},
        "gt_object_names": {L: by_label[L]["name"] for L in labels},
        "difficulty": gt.get("difficulty"),
        "triple_tier": gt.get("triple_tier"),
        "target_occlusions": gt.get("occlusions"),
        "n_labels_hidden_t0": gt.get("n_labels_hidden_t0"),
        "largest_runnerup_ratio": gt.get("largest_runnerup_ratio"),
        "largest_letter": gt.get("largest_letter"),
        "smallest_letter": gt.get("smallest_letter"),
        "agentic_constraint": AGENTIC_CONSTRAINT,
    }


def score(pred_letter, q: dict) -> bool:
    """Letter-only scoring, as in the benchmark's relative-size task."""
    return bool(pred_letter) and str(pred_letter).strip().upper()[:1] == \
        q["gt_answer"].upper()


# ---------------------------------------------------------------------------
# Type-aware system prompts
# ---------------------------------------------------------------------------

_STALE_SINGLEPASS = ("For same_attribute and relative_size questions, objects A "
                     "and B are identified by normalized [row, col] points given "
                     "in the prompt (the image itself is unmarked); locate each "
                     "object at its point and reason about it.")

_MARKED_SINGLEPASS_LINE = (
    "Three objects are identified by badges PAINTED ONTO them: a white circle "
    "A, square B, and triangle C. No coordinates are given; find badges in the "
    "image yourself. A badge may be unreadable where its object is covered.")

REPORT_BADGES_DOC = """
== MARKED-COMPARISON BADGES ==
Three objects carry painted badges: circle A, square B, triangle C. No badge
coordinates are provided. Locating them can help identify the comparison
objects, but it is optional: you may answer whenever you have enough evidence.

6. report_badges(A, B, C)
   Optionally record currently readable badge locations as normalised [row,col]
   points on [0,1000], or null for any unreadable badge.
   Example: {"tool":"report_badges", "A":[412,233], "B":null,
             "C":[690,718], "reasoning":"B remains covered."}
   - FREE: does not consume manipulation budget.
   - This is a diagnostic note, NOT an answer prerequisite. report_badges is
     optional and answer_question is always permitted.
"""


def patch_agentic_system_prompt(sp: str, perception_only: bool = False,
                                include_badge_report: bool = False) -> str:
    """Make the shared tool prompt correct for marked three-way comparison."""
    stale = ("- For same_attribute and relative_size questions, two objects are "
             "designated A and B. Each has a normalized [row, col] centroid on a "
             "[0, 1000] grid (row=0 top, col=0 left). A blank point (`[ ,  ]`) means "
             "the target is severely cluttered/occluded; clear obstructors and the "
             "point will reappear automatically. Do not commit to an answer until "
             "both A and B have non-blank points.")
    sp = sp.replace(stale, "")
    # The generic prompt asserts images are never annotated — false for the
    # marked comparison task, where three identity badges are painted ON the
    # objects in the image. Correct that line so the agent trusts the badges.
    stale_never_annotated = (
        "Images are NEVER annotated; all perception results, object IDs, and "
        "A/B target designations are given as text inside the user turn.")
    badge_note = (
        "The only marks ever drawn on an image are the three identity badges "
        "for THIS task: a circle labels object A, a square labels object B, and "
        "a triangle labels object C, painted directly on the objects. Badges "
        "identify objects; they are NOT a size cue. All other perception "
        "results and object IDs are given as text inside the user turn.")
    sp = sp.replace(stale_never_annotated, badge_note)
    # The generic benchmark prompt normally tells the agent never to remove a
    # question target. For this manipulation benchmark that would be an extra
    # privileged safeguard: if it removes a labelled object, it is the agent's
    # own tracking mistake. Remove the generic restriction only for this type.
    sp = sp.replace("- Do NOT remove objects the question asks about.\n", "")
    full_tools = ('- "tool": one of "perceive_scene", "perceive_find_object", '
                  '"grasp_and_remove", "push", "answer_question"')
    report_tool = ', "report_badges"' if include_badge_report else ''
    if perception_only:
        replacement = ('- "tool": one of "perceive_scene", "perceive_find_object"'
                       + report_tool + ', "answer_question"')
        sp = sp.replace(full_tools, replacement)
        sp += ("\n\n== PERCEPTION-ONLY CONDITION ==\n"
               "Manipulation is unavailable in this condition. Use perception "
               "tools or answer from the unchanged scene; do not request push "
               "or grasp_and_remove.\n")
    else:
        replacement = ('- "tool": one of "perceive_scene", "perceive_find_object", '
                       '"grasp_and_remove", "push"' + report_tool +
                       ', "answer_question"')
        sp = sp.replace(full_tools, replacement)
    if include_badge_report:
        sp += "\n" + REPORT_BADGES_DOC
    return sp.rstrip()


def patch_singlepass_system_prompt(sp: str) -> str:
    return sp.replace(_STALE_SINGLEPASS, _MARKED_SINGLEPASS_LINE)
