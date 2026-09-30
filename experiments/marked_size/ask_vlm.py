#!/usr/bin/env python3
"""Initial experiment: can a VLM answer the marked size question?

Runs the single-pass (no tools) baseline on each generated scene under two
conditions:

  scene     - the real image, where at least one badge is buried under clutter
  revealed  - the same scene with the clutter parked away (ORACLE: both badges
              readable)

The gap between the two is the headroom that manipulation is supposed to
unlock.  If ``revealed`` is high and ``scene`` is near chance, the task is
well-formed: the answer is knowable, but only after de-occlusion.

No simulator is launched here -- it reads the PNGs written by make_scenes.py,
so you can re-ask with different models cheaply.

Usage:
    python -m experiments.marked_size.ask_vlm \
        --model gcp/google/gemini-3.1-pro-preview
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"), override=False)
except ImportError:
    pass

from models import call_model, parse_vlm_response            # noqa: E402
from experiments.marked_size import question as Q            # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def ask_one(img_rgb, q, model, ask_ratio=True):
    prompt = Q.build_singlepass_prompt(q, ask_ratio=ask_ratio)
    raw = call_model(images=[img_rgb], prompt=prompt,
                     system_prompt=Q.SINGLEPASS_SYSTEM_PROMPT,
                     model_name=model, temp=0.0, n_tokens=2048)
    parsed = parse_vlm_response(raw) or {}
    return {
        "answer": parsed.get("answer"),
        "ratio": parsed.get("ratio"),
        "labels_visible": parsed.get("labels_visible"),
        "reasoning": (parsed.get("reasoning") or "")[:300],
        "raw": raw[:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str,
                    default="gcp/google/gemini-3.1-pro-preview")
    ap.add_argument("--conditions", type=str, default="scene,revealed")
    ap.add_argument("--no_ratio", action="store_true")
    ap.add_argument("--sites", type=str,
                    default=os.path.join(OUT, "sites.json"))
    args = ap.parse_args()

    doc = json.load(open(args.sites))
    scenes = doc["scenes"]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"model      : {args.model}")
    print(f"scenes     : {len(scenes)}")
    print(f"conditions : {conditions}\n")

    results = []
    for r in scenes:
        pair = r["labels"]
        q = Q.build_question(pair, r["gt"])
        for cond in conditions:
            path = os.path.join(OUT, f"{r['tag']}_{cond}.png")
            if not os.path.isfile(path):
                print(f"  ! missing {path}")
                continue
            bgr = cv2.imread(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                out = ask_one(rgb, q, args.model, ask_ratio=not args.no_ratio)
            except Exception as e:
                print(f"  {r['tag']} {cond:9s} ERROR {type(e).__name__}: {e}")
                results.append({"tag": r["tag"], "condition": cond,
                                "error": str(e)[:200]})
                continue
            ok = Q.score(out["answer"], q)
            hidden = q["n_labels_hidden_t0"]
            print(f"  {r['tag']} {cond:9s} pred={str(out['answer'])[:2]:2s} "
                  f"gt={q['gt_answer']}  {'OK ' if ok else 'X  '} "
                  f"(hidden_t0={hidden}, gt_ratio={q['gt_ratio']}, "
                  f"pred_ratio={out['ratio']})")
            results.append({
                "tag": r["tag"], "condition": cond, "correct": bool(ok),
                "pred": out["answer"], "gt": q["gt_answer"],
                "gt_ratio": q["gt_ratio"], "pred_ratio": out["ratio"],
                "labels_visible_pred": out["labels_visible"],
                "difficulty": q["difficulty"],
                "n_labels_hidden_t0": hidden,
                "reasoning": out["reasoning"],
            })

    print("\n" + "=" * 62)
    print(f"{'condition':12s} {'n':>4s} {'correct':>8s} {'accuracy':>9s}")
    print("-" * 62)
    for cond in conditions:
        rs = [x for x in results if x.get("condition") == cond and "correct" in x]
        if not rs:
            continue
        n = len(rs)
        c = sum(x["correct"] for x in rs)
        print(f"{cond:12s} {n:4d} {c:8d} {100.0*c/n:8.1f}%")
    # Split by how many labels were buried at t0
    print("-" * 62)
    for cond in conditions:
        for diff in ("one_hidden", "both_hidden"):
            rs = [x for x in results if x.get("condition") == cond
                  and x.get("difficulty") == diff and "correct" in x]
            if rs:
                c = sum(x["correct"] for x in rs)
                print(f"{cond:12s} {diff:12s} {len(rs):3d} "
                      f"{100.0*c/len(rs):6.1f}%")
    print("=" * 62)

    path = os.path.join(OUT, "vlm_results.json")
    with open(path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
