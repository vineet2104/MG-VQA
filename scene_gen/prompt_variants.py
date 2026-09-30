"""Prompt variant registry for VLM robustness testing.

Each question type has a set of semantically equivalent template variants
tagged by category (synonym, inversion, casual, typo, etc.). The canonical
template matches the original hardcoded question in demo_vlm.py.

Usage:
    from scene_gen.prompt_variants import get_template, list_variants

    # Get a specific variant
    tmpl = get_template("under_object", "uo_below")
    question = tmpl.format(cover="dish towel")

    # Get canonical (default)
    tmpl = get_template("under_object")

    # List all variants for a type
    for v in list_variants("under_object"):
        print(v["id"], v["cat"], v["template"])
"""
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Variant definitions per question type
# ---------------------------------------------------------------------------

VARIANTS: Dict[str, List[dict]] = {}

# ── under_object ──────────────────────────────────────────────────────────

VARIANTS["under_object"] = [
    # Canonical (matches demo_vlm.py exactly)
    {"id": "uo_canonical", "cat": "canonical",
     "template": "What object is under the {cover}?"},

    # Synonym substitutions
    {"id": "uo_below", "cat": "synonym",
     "template": "What object is below the {cover}?"},
    {"id": "uo_beneath", "cat": "synonym",
     "template": "What object is beneath the {cover}?"},
    {"id": "uo_underneath", "cat": "synonym",
     "template": "What object is underneath the {cover}?"},

    # Inversions (reframed perspective)
    {"id": "uo_on_top_of", "cat": "inversion",
     "template": "The {cover} is on top of what object?"},
    {"id": "uo_covering", "cat": "inversion",
     "template": "What is the {cover} covering?"},
    {"id": "uo_hiding", "cat": "inversion",
     "template": "What object is the {cover} hiding?"},

    # Passive voice
    {"id": "uo_hidden_by", "cat": "passive",
     "template": "What is hidden by the {cover}?"},
    {"id": "uo_obscured", "cat": "passive",
     "template": "What object is obscured by the {cover}?"},

    # Casual / natural
    {"id": "uo_look_under", "cat": "casual",
     "template": "look under the {cover}. what object?"},
    {"id": "uo_whats_under", "cat": "casual",
     "template": "what's under the {cover}?"},
    {"id": "uo_can_you_tell", "cat": "casual",
     "template": "can you tell me what's under the {cover}?"},

    # Imperative / command
    {"id": "uo_identify", "cat": "imperative",
     "template": "Identify the object under the {cover}."},
    {"id": "uo_name", "cat": "imperative",
     "template": "Name the object that is under the {cover}."},

    # Verbose / over-specified
    {"id": "uo_lift", "cat": "verbose",
     "template": "If you were to lift the {cover}, what object would you find underneath it?"},
    {"id": "uo_remove", "cat": "verbose",
     "template": "Suppose the {cover} were removed. What object is it currently sitting on top of?"},

    # Terse / minimal
    {"id": "uo_terse_eq", "cat": "terse",
     "template": "Under {cover} = ?"},
    {"id": "uo_terse_arrow", "cat": "terse",
     "template": "{cover} \u2192 what's below?"},

    # Typos (ecologically valid — real user mistakes)
    {"id": "uo_typo_object", "cat": "typo",
     "template": "What obejct is under the {cover}?"},
    {"id": "uo_typo_under", "cat": "typo",
     "template": "What object is udner the {cover}?"},
    {"id": "uo_typo_what", "cat": "typo",
     "template": "Waht object is under the {cover}?"},

    # Capitalization
    {"id": "uo_allcaps", "cat": "caps",
     "template": "WHAT OBJECT IS UNDER THE {COVER}?"},
    {"id": "uo_lowercase", "cat": "caps",
     "template": "what object is under the {cover}?"},

    # Punctuation
    {"id": "uo_no_question_mark", "cat": "punctuation",
     "template": "What object is under the {cover}"},
    {"id": "uo_double_question", "cat": "punctuation",
     "template": "What object is under the {cover}??"},
    {"id": "uo_ellipsis", "cat": "punctuation",
     "template": "What object is under the {cover}...?"},
]

# ── existence ─────────────────────────────────────────────────────────────

VARIANTS["existence"] = [
    {"id": "ex_canonical", "cat": "canonical",
     "template": "Is there a {queried} on the table?"},
    # Synonym
    {"id": "ex_can_you_see", "cat": "synonym",
     "template": "Can you see a {queried} on the table?"},
    {"id": "ex_present", "cat": "synonym",
     "template": "Is a {queried} present on the table?"},
    {"id": "ex_visible", "cat": "synonym",
     "template": "Is a {queried} visible on the table?"},
    # Inversion
    {"id": "ex_table_contain", "cat": "inversion",
     "template": "Does the table contain a {queried}?"},
    {"id": "ex_among", "cat": "inversion",
     "template": "Is a {queried} among the objects on the table?"},
    # Passive
    {"id": "ex_placed", "cat": "passive",
     "template": "Has a {queried} been placed on the table?"},
    # Casual
    {"id": "ex_spot", "cat": "casual",
     "template": "do you spot a {queried} on the table?"},
    {"id": "ex_see_any", "cat": "casual",
     "template": "see any {queried} on the table?"},
    # Imperative
    {"id": "ex_check", "cat": "imperative",
     "template": "Check if there is a {queried} on the table."},
    {"id": "ex_confirm", "cat": "imperative",
     "template": "Confirm whether a {queried} is on the table."},
    # Verbose
    {"id": "ex_anywhere", "cat": "verbose",
     "template": "Is there a {queried} anywhere on the table surface?"},
    {"id": "ex_looking_at", "cat": "verbose",
     "template": "Looking at the table, can you determine if a {queried} is present?"},
    # Terse
    {"id": "ex_terse", "cat": "terse",
     "template": "{queried} on table?"},
    # Typo
    {"id": "ex_typo", "cat": "typo",
     "template": "Is there a {queried} on teh table?"},
    {"id": "ex_typo2", "cat": "typo",
     "template": "Is there a {queried} on the tabel?"},
    # Caps
    {"id": "ex_allcaps", "cat": "caps",
     "template": "IS THERE A {QUERIED} ON THE TABLE?"},
    {"id": "ex_lowercase", "cat": "caps",
     "template": "is there a {queried} on the table?"},
    # Punctuation
    {"id": "ex_no_question", "cat": "punctuation",
     "template": "Is there a {queried} on the table"},
    {"id": "ex_ellipsis", "cat": "punctuation",
     "template": "Is there a {queried} on the table...?"},
]

# ── counting ──────────────────────────────────────────────────────────────

VARIANTS["counting"] = [
    {"id": "ct_canonical", "cat": "canonical",
     "template": "How many {queried} objects are on the table?"},
    # Synonym
    {"id": "ct_count", "cat": "synonym",
     "template": "Count the number of {queried} objects on the table."},
    {"id": "ct_quantity", "cat": "synonym",
     "template": "What quantity of {queried} objects are on the table?"},
    # Inversion
    {"id": "ct_table_has", "cat": "inversion",
     "template": "The table has how many {queried} objects?"},
    # Casual
    {"id": "ct_how_many_see", "cat": "casual",
     "template": "how many {queried} things do you see on the table?"},
    {"id": "ct_got", "cat": "casual",
     "template": "how many {queried} objects we got on the table?"},
    # Imperative
    {"id": "ct_tell_me", "cat": "imperative",
     "template": "Tell me how many {queried} objects are on the table."},
    # Verbose
    {"id": "ct_total", "cat": "verbose",
     "template": "What is the total number of {queried} objects that are on the table?"},
    {"id": "ct_examine", "cat": "verbose",
     "template": "Examine the table carefully and count how many {queried} objects you can find."},
    # Terse
    {"id": "ct_terse", "cat": "terse",
     "template": "# of {queried} on table?"},
    # Typo
    {"id": "ct_typo", "cat": "typo",
     "template": "How manu {queried} objects are on the table?"},
    {"id": "ct_typo2", "cat": "typo",
     "template": "How many {queried} obejcts are on the table?"},
    # Caps
    {"id": "ct_allcaps", "cat": "caps",
     "template": "HOW MANY {QUERIED} OBJECTS ARE ON THE TABLE?"},
    {"id": "ct_lowercase", "cat": "caps",
     "template": "how many {queried} objects are on the table?"},
    # Punctuation
    {"id": "ct_no_question", "cat": "punctuation",
     "template": "How many {queried} objects are on the table"},
]

# ── same_attribute (A/B markers — object names never in question) ─────────

VARIANTS["same_attribute"] = [
    # Canonical (matches questions.py exactly)
    {"id": "sa_canonical", "cat": "canonical",
     "template": "Two objects in the scene are marked A and B. Do they have the same color?"},
    # Synonym
    {"id": "sa_match", "cat": "synonym",
     "template": "Two objects are labelled A and B. Do they match in color?"},
    {"id": "sa_identical", "cat": "synonym",
     "template": "Objects A and B are marked in the scene. Is their color identical?"},
    {"id": "sa_share", "cat": "synonym",
     "template": "Do the objects marked A and B share the same color?"},
    # Inversion
    {"id": "sa_differ", "cat": "inversion",
     "template": "Two objects marked A and B are in the scene. Do they differ in color?"},
    {"id": "sa_different", "cat": "inversion",
     "template": "Are objects A and B different colors?"},
    # Passive
    {"id": "sa_colored_same", "cat": "passive",
     "template": "Are the objects marked A and B colored the same?"},
    {"id": "sa_observed", "cat": "passive",
     "template": "Can the objects labelled A and B be observed to have the same color?"},
    # Casual
    {"id": "sa_casual", "cat": "casual",
     "template": "are A and B the same color?"},
    {"id": "sa_same_q", "cat": "casual",
     "template": "same color — A and B?"},
    {"id": "sa_look_at", "cat": "casual",
     "template": "look at the objects marked A and B. same color or not?"},
    # Imperative
    {"id": "sa_compare", "cat": "imperative",
     "template": "Compare the colors of object A and object B."},
    {"id": "sa_determine", "cat": "imperative",
     "template": "Determine whether the two objects marked A and B have the same color."},
    # Verbose
    {"id": "sa_looking", "cat": "verbose",
     "template": "Looking at the two objects marked A and B in the scene, would you say they are the same color?"},
    {"id": "sa_examine", "cat": "verbose",
     "template": "Examine the objects labelled A and B carefully. Do they appear to be the same color?"},
    # Terse
    {"id": "sa_terse", "cat": "terse",
     "template": "A and B — same color?"},
    {"id": "sa_terse_eq", "cat": "terse",
     "template": "color(A) == color(B)?"},
    # Typo
    {"id": "sa_typo", "cat": "typo",
     "template": "Two objects marked A and B. Do they have the same color?"},
    {"id": "sa_typo2", "cat": "typo",
     "template": "Do the objects marked A and B ahve the same color?"},
    # Caps
    {"id": "sa_allcaps", "cat": "caps",
     "template": "TWO OBJECTS IN THE SCENE ARE MARKED A AND B. DO THEY HAVE THE SAME COLOR?"},
    {"id": "sa_lowercase", "cat": "caps",
     "template": "two objects in the scene are marked a and b. do they have the same color?"},
    # Punctuation
    {"id": "sa_no_question", "cat": "punctuation",
     "template": "Two objects in the scene are marked A and B. Do they have the same color"},
    {"id": "sa_ellipsis", "cat": "punctuation",
     "template": "Objects marked A and B... same color?"},
]

# ── relative_size (A/B markers — object names never in question) ──────────
# Note: the ratio sub-question is appended by _apply_random_variant, not here.

VARIANTS["relative_size"] = [
    # Canonical (matches questions.py — only the "which is larger" part)
    {"id": "rs_canonical", "cat": "canonical",
     "template": "Two objects in the scene are marked A and B. Which one is larger?"},
    # Synonym
    {"id": "rs_bigger", "cat": "synonym",
     "template": "Two objects are labelled A and B. Which one is bigger?"},
    {"id": "rs_more_space", "cat": "synonym",
     "template": "Objects A and B are marked in the scene. Which takes up more space?"},
    {"id": "rs_greater_volume", "cat": "synonym",
     "template": "Of the two objects marked A and B, which has greater volume?"},
    # Inversion
    {"id": "rs_smaller_inv", "cat": "inversion",
     "template": "Two objects marked A and B. Which one is smaller?"},
    {"id": "rs_less_space", "cat": "inversion",
     "template": "Between objects A and B, which occupies less space?"},
    # Passive
    {"id": "rs_considered", "cat": "passive",
     "template": "Between the objects marked A and B, which would be considered larger?"},
    # Casual
    {"id": "rs_casual", "cat": "casual",
     "template": "A or B — which is bigger?"},
    {"id": "rs_which_one", "cat": "casual",
     "template": "look at objects A and B. which one's larger?"},
    # Imperative
    {"id": "rs_identify", "cat": "imperative",
     "template": "Identify which of the two objects marked A and B is larger."},
    {"id": "rs_tell_me", "cat": "imperative",
     "template": "Tell me which is larger, object A or object B."},
    # Verbose
    {"id": "rs_compare", "cat": "verbose",
     "template": "Comparing the two objects marked A and B in the scene, which one appears to be physically larger?"},
    {"id": "rs_estimate", "cat": "verbose",
     "template": "Carefully examine objects A and B. Estimate which one is larger in overall size."},
    # Terse
    {"id": "rs_terse", "cat": "terse",
     "template": "Larger: A or B?"},
    # Typo
    {"id": "rs_typo", "cat": "typo",
     "template": "Two objects maked A and B. Which is lager?"},
    {"id": "rs_typo2", "cat": "typo",
     "template": "Wich of the objects marked A and B is larger?"},
    # Caps
    {"id": "rs_allcaps", "cat": "caps",
     "template": "TWO OBJECTS MARKED A AND B. WHICH ONE IS LARGER?"},
    {"id": "rs_lowercase", "cat": "caps",
     "template": "two objects marked a and b. which one is larger?"},
    # Punctuation
    {"id": "rs_no_question", "cat": "punctuation",
     "template": "Two objects marked A and B. Which one is larger"},
]

# ── interactive_counting ──────────────────────────────────────────────────

VARIANTS["interactive_counting"] = [
    # Canonical (matches questions.py)
    {"id": "ic_canonical", "cat": "canonical",
     "template": "Remove exactly {x} object(s) from the table, then count how many objects remain on the table."},
    # Synonym
    {"id": "ic_take_away", "cat": "synonym",
     "template": "Take away {x} object(s) from the table. How many objects are left?"},
    {"id": "ic_clear", "cat": "synonym",
     "template": "Clear {x} object(s) off the table and count the remaining objects."},
    # Inversion
    {"id": "ic_after", "cat": "inversion",
     "template": "After removing {x} object(s), how many objects will remain on the table?"},
    {"id": "ic_how_many_left", "cat": "inversion",
     "template": "How many objects are left on the table if you remove {x} of them?"},
    # Passive
    {"id": "ic_once_removed", "cat": "passive",
     "template": "Once {x} object(s) have been removed from the table, how many remain?"},
    # Casual
    {"id": "ic_grab", "cat": "casual",
     "template": "grab {x} objects off the table. how many are still there?"},
    {"id": "ic_pick_up", "cat": "casual",
     "template": "pick up {x} things from the table, then tell me how many are left."},
    # Imperative
    {"id": "ic_remove_count", "cat": "imperative",
     "template": "Remove {x} object(s) from the table. Report the number of remaining objects."},
    {"id": "ic_first_then", "cat": "imperative",
     "template": "First remove {x} object(s) from the table. Then count what remains."},
    # Verbose
    {"id": "ic_carefully", "cat": "verbose",
     "template": "Carefully remove exactly {x} object(s) from the table, then count how many objects are still on the table."},
    {"id": "ic_task", "cat": "verbose",
     "template": "Your task: remove {x} object(s) from the table and determine the total number of objects that remain."},
    # Terse
    {"id": "ic_terse", "cat": "terse",
     "template": "Remove {x}, count remaining."},
    # Typo
    {"id": "ic_typo", "cat": "typo",
     "template": "Remov exactly {x} object(s) form the table, then count how many remain."},
    # Caps
    {"id": "ic_allcaps", "cat": "caps",
     "template": "REMOVE EXACTLY {X} OBJECT(S) FROM THE TABLE, THEN COUNT HOW MANY REMAIN."},
    {"id": "ic_lowercase", "cat": "caps",
     "template": "remove exactly {x} object(s) from the table, then count how many objects remain on the table."},
    # Punctuation
    {"id": "ic_no_period", "cat": "punctuation",
     "template": "Remove exactly {x} object(s) from the table, then count how many objects remain on the table"},
]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def get_template(question_type: str, variant_id: Optional[str] = None) -> str:
    """Get a question template by type and optional variant ID.

    If variant_id is None, returns the canonical template.
    """
    variants = VARIANTS.get(question_type, [])
    if not variants:
        raise KeyError(f"No variants defined for question type: {question_type}")

    if variant_id is None:
        # Return canonical
        for v in variants:
            if v["cat"] == "canonical":
                return v["template"]
        return variants[0]["template"]

    for v in variants:
        if v["id"] == variant_id:
            return v["template"]

    raise KeyError(f"Unknown variant '{variant_id}' for type '{question_type}'. "
                   f"Available: {[v['id'] for v in variants]}")


def list_variants(question_type: Optional[str] = None) -> List[dict]:
    """List all variants, optionally filtered by question type."""
    if question_type:
        return VARIANTS.get(question_type, [])
    all_variants = []
    for qt, vs in VARIANTS.items():
        for v in vs:
            all_variants.append({"question_type": qt, **v})
    return all_variants


def list_variant_ids(question_type: str) -> List[str]:
    """List all variant IDs for a question type."""
    return [v["id"] for v in VARIANTS.get(question_type, [])]


def get_variant_info(variant_id: str) -> Optional[dict]:
    """Look up a variant by ID across all question types."""
    for qt, vs in VARIANTS.items():
        for v in vs:
            if v["id"] == variant_id:
                return {"question_type": qt, **v}
    return None
