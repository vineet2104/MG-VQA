"""
orchestrator.summary — Post-run summary generation (summary.json).

Generates aggregate accuracy metrics, per-question-type breakdown,
manipulation statistics, and failure-reason analysis from results.jsonl.
"""

import os
import json
from collections import Counter


def generate_summary(save_dir: str, records: list) -> None:
    """Generate summary.json from merged records.

    SCORING NOTE — relative_size:
      For relative_size questions the *primary* correctness signal is
      ``letter_correct`` (did the model pick the right A/B object?).
      The ``is_correct`` field stored on each record is the strict
      conjunction of letter AND ratio (within ±10%), which most analyses
      do not want. This summary uses ``letter_correct`` for
      relative_size and ``is_correct`` for every other question type.
      The strict ``is_correct`` count is still reported as a side
      metric (``relative_size_strict_correct``) so nothing is lost.

    Overall and per-type accuracy use correct / answered only. Non-answer
    failures (including max steps and perception-loop termination), infrastructure
    errors, skipped episodes and saved snapshots are reported separately and
    never included in an answer-accuracy denominator.
    """
    if not records:
        return

    def _is_max_steps(r):
        reason = (r.get("fail_reason") or "").lower()
        return r.get("status") == "fail" and "maximum vlm steps exceeded" in reason

    def _is_infra(r):
        return r.get("status") not in (
            "answered", "skipped", "snapshot_saved", "fail", "perception_loop_failure")

    def _letter_score(r):
        """Letter-only correctness: relative_size uses letter_correct,
        every other type uses is_correct."""
        from mgvqa.metrics import is_answer_correct
        return is_answer_correct(r)

    answered = [r for r in records if r.get("status") == "answered"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    max_steps = [r for r in records if _is_max_steps(r)]
    infra = [r for r in records if _is_infra(r)]

    n_total = len(records) - len(skipped)            # legacy denominator
    n_answered = len(answered)
    failed = [r for r in records if r.get("status") in ("fail", "perception_loop_failure")]
    n_correct = sum(1 for r in answered if _letter_score(r))

    accuracy_answered = n_correct / n_answered if n_answered else 0.0
    from mgvqa.metrics import answer_metrics
    answer_stats = answer_metrics(records)
    type_stats = answer_stats["per_type"]

    # Relative-size sub-metrics: letter-only is primary; ratio (within
    # ±10% of GT) and strict (letter AND ratio) are diagnostic only.
    rs_answered = [r for r in answered if r.get("question_type") == "relative_size"]
    n_rs_answered = len(rs_answered)
    n_rs_letter = sum(1 for r in rs_answered
                      if r.get("letter_correct", r.get("is_correct")))
    n_rs_ratio = sum(1 for r in rs_answered if r.get("ratio_correct"))
    rs_strict_correct = sum(1 for r in rs_answered if r.get("is_correct"))

    # Skip breakdown by (type, reason) AND a flat by-reason rollup so
    # analyses don't have to re-aggregate.
    skip_stats = {}
    skip_reason_totals = {}
    for r in skipped:
        qt = r.get("question_type", "unknown")
        reason = r.get("skip_reason", "unknown")
        key = f"{qt}/{reason}"
        skip_stats[key] = skip_stats.get(key, 0) + 1
        skip_reason_totals[reason] = skip_reason_totals.get(reason, 0) + 1

    # Infra-failure breakdown by coarse class (mirrors
    # scripts/failure_analysis.py).
    def _classify_infra(r):
        reason = " ".join([
            r.get("fail_reason") or "",
            r.get("error") or "",
        ]).lower()
        if "readtimeout" in reason or "read timeout" in reason:
            return "api_timeout"
        if "vlm parse error" in reason or "could not extract json" in reason:
            return "vlm_parse_error"
        if "cannot reshape tensor of 0 elements" in reason:
            return "graspgen_zero_pc"
        if "cuda" in reason and "graspgen" in reason:
            return "graspgen_cuda"
        return "other"

    infra_stats = {}
    for r in infra:
        cls = _classify_infra(r)
        qt = r.get("question_type", "unknown")
        key = f"{qt}/{cls}"
        infra_stats[key] = infra_stats.get(key, 0) + 1

    # Manipulation stats — accumulate over every episode that ran the
    # robot at least once (answered + max_steps + infra).  Skipped scenes
    # never spawned objects so they contribute nothing.
    completed = [r for r in records if r.get("status") != "skipped"]
    total_grasps = sum(r.get("grasp_attempts", 0) for r in completed)
    grasp_success = sum(r.get("grasp_successes", 0) for r in completed)
    total_pushes = sum(r.get("push_attempts", 0) for r in completed)
    push_success = sum(r.get("push_successes", 0) for r in completed)

    summary = {
        # ---- Episode counts ----
        "n_episodes_total":     len(records),
        "n_episodes_skipped":   len(skipped),
        "n_episodes_completed": n_total,           # legacy: total - skipped
        "n_episodes_answered":  n_answered,
        "n_episodes_max_steps": len(max_steps),
        "n_episodes_infra":     len(infra),
        "n_episodes_failed":    len(failed),
        "n_episodes_perception_loop_failure": sum(
            r.get("status") == "perception_loop_failure" for r in failed),
        "status_counts": answer_stats["status_counts"],
        "non_answer_failures_by_reason": answer_stats["non_answer_failures_by_reason"],
        "n_correct":            n_correct,
        # ---- Primary accuracy: correct / answered ----
        "accuracy":             round(accuracy_answered, 3),
        "accuracy_answered":    round(accuracy_answered, 3),
        "_accuracy_notes": {
            "accuracy":           "n_correct / n_answered  [PRIMARY — only episodes where VLM produced an answer]",
            "accuracy_answered":  "n_correct / n_answered  [same as 'accuracy']",
            "per_type": "correct / answered; total is inventory only; non-answer failures excluded",
            "removed_legacy_metric": "accuracy_attempted removed: non-answer failures are reported separately",
            "scoring":            "relative_size uses letter_correct (primary). All other types use is_correct. Strict letter-AND-ratio count is reported as relative_size_detail.strict_correct.",
        },
        "per_type": type_stats,
        # Diagnostic break-down for relative_size: letter is the primary
        # signal; ratio (within ±10% of GT) is reported separately because
        # volume estimation from a top-down view is unreliable; strict =
        # legacy is_correct (letter AND ratio).
        "relative_size_detail": {
            "n_answered":       n_rs_answered,
            "letter_correct":   n_rs_letter,
            "letter_accuracy":  round(n_rs_letter / n_rs_answered, 3) if n_rs_answered else 0.0,
            "ratio_correct":    n_rs_ratio,
            "ratio_accuracy":   round(n_rs_ratio / n_rs_answered, 3) if n_rs_answered else 0.0,
            "strict_correct":   rs_strict_correct,
            "strict_accuracy":  round(rs_strict_correct / n_rs_answered, 3) if n_rs_answered else 0.0,
            "_note": "letter_accuracy is used for headline per_type accuracy; ratio and strict are diagnostic only.",
        },
        "skipped": skip_stats,
        "skipped_by_reason": skip_reason_totals,
        "infra_errors": infra_stats,
        "manipulation": {
            "grasps": {
                "attempts": total_grasps,
                "successes": grasp_success,
                "success_rate": round(grasp_success / total_grasps, 3)
                                if total_grasps else 0.0,
            },
            "pushes": {
                "attempts": total_pushes,
                "successes": push_success,
                "success_rate": round(push_success / total_pushes, 3)
                                if total_pushes else 0.0,
            },
        },
    }

    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {summary_path}")
    print(f"  ┌─────────────────────────────────────────────────────────")
    print(f"  │ ACCURACY: {n_correct}/{n_answered} = {accuracy_answered:.1%}"
          f"  (correct / answered)")
    print(f"  │ Scoring : letter-only for relative_size, is_correct for others")
    print(f"  │")
    n_snapshot = sum(1 for r in records
                     if r.get("status") == "snapshot_saved")
    if n_snapshot:
        # Snapshot-generation run: no answers, just saved scenes.
        print(f"  │ Episodes: {len(records)} total"
              f" → {n_snapshot} snapshot_saved"
              f" → {len(skipped)} skipped"
              f" + {len(infra)} infra-error")
    else:
        print(f"  │ Episodes: {len(records)} total"
              f" → {len(skipped)} skipped"
              f" → {n_answered} answered"
              f" + {len(failed)} non-answer failures ({len(max_steps)} max-steps)"
              f" + {len(infra)} infra-error")
    if skipped:
        # Flat per-reason rollup with human-readable descriptions.
        reason_descriptions = {
            "not_in_subset":          "excluded from benchmark via --episode_subset",
            "few_shot_exemplar":      "reserved as a few-shot exemplar episode",
            "insufficient_occlusion": "scene did not yield enough occlusion to be challenging",
            "no_target_object":       "validator could not place a usable target object",
            "ab_visibility":          "A/B-target visibility check failed",
            "spawn_error":            "PyBullet spawn / scene-build failure",
        }
        for reason, count in sorted(
                skip_reason_totals.items(), key=lambda kv: -kv[1]):
            desc = reason_descriptions.get(reason, "")
            tail = f"  ({desc})" if desc else ""
            print(f"  │   • {count} skipped: {reason}{tail}")
    for reason, count in sorted(answer_stats["non_answer_failures_by_reason"].items()):
        print(f"  │   • {count} non-answer: {reason}")
    if infra:
        for key, count in sorted(infra_stats.items()):
            print(f"  │   • {count} infra: {key}")
    print(f"  │")
    print(f"  │ Per type:")
    for qt in sorted(type_stats.keys()):
        s = type_stats[qt]
        print(f"  │   {qt:<22s} {s['correct']:>2}/{s['answered']:<2}"
              f"  = {s['accuracy']:.1%}  (non-answer: {s['non_answer']})")
    if n_rs_answered:
        print(f"  │")
        print(f"  │ Relative size detail: letter={n_rs_letter}/{n_rs_answered}"
              f"  ratio={n_rs_ratio}/{n_rs_answered}"
              f"  strict={rs_strict_correct}/{n_rs_answered}  (diagnostic only)")
    print(f"  └─────────────────────────────────────────────────────────")


