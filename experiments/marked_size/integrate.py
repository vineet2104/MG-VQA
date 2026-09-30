"""Three-object marked-size episode construction.

Badges are baked into scratch texture copies and pushed to both PyBullet and
Blender.  Candidate triples are counterbalanced by the occlusion rank of the
true largest object, so choosing the least-visible target is chance (1/3) under
macro averaging across the three ranks.
"""

import itertools
import os
from typing import Dict, Optional, Tuple

import numpy as np

from experiments.marked_size import config as C
from experiments.marked_size import marker_lib as ML
from experiments.marked_size import question as Q


_LABELS = tuple(C.LABELS)


def _volume(site: dict, gt_lookup: Dict) -> float:
    size = (gt_lookup.get(site["name"]) or {}).get("size_cm") or {}
    return max(float(size.get("length", 0)) * float(size.get("width", 0)) *
               float(size.get("height", 0)), 1e-9)


def _eligible(demo, folder_by_name, asset_path, occ_stats, seg):
    """Return objects with at least one in-frame, self-visible badge site."""
    cam = demo.env.camera
    cam_eye = list(getattr(cam, "cam_pos", [0.05, -0.52, 1.9]))
    occl = occ_stats.get("occlusion", {})
    out = []
    for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
        folder = folder_by_name.get(oname)
        if not folder or os.path.basename(folder) in C.BADGE_ASSET_DENYLIST:
            continue
        obj_dir = os.path.join(asset_path, folder)
        mesh, tex = ML.load_mesh(obj_dir), ML.texture_path(obj_dir)
        if mesh is None or tex is None:
            continue
        # Use visual, rather than inertial, pose. This is the transform that
        # Blender applies when rendering the corresponding texture-mapped mesh.
        pos, quat = demo.env._get_visual_frame(oid, -1, demo.env.physicsClient)
        offsets = demo.env._get_visual_origin_offsets(oid)

        def _in_frame(wp):
            r, c = ML.project_to_pixel(cam, wp)
            m = C.FRAME_MARGIN_PX
            return m <= r < cam.height - m and m <= c < cam.width - m

        sites = ML.pick_marker_site(mesh, pos, quat, offsets.get(-1),
                                    validate=_in_frame, cam_eye=cam_eye,
                                    max_sites=6)
        if not sites:
            continue
        # Prefer an initially identifiable site if there is one. The object may
        # still be occluded; badge visibility is a diagnostic, never a filter.
        scored = []
        for site in sites:
            vis = ML.badge_visible_frac(cam, seg, site["world_point"],
                                        site["badge_cm_actual"], oid)
            scored.append((vis["frac"], site, vis))
        _, site, vis = max(scored, key=lambda x: x[0])
        site.update(obj_id=oid, name=oname, folder=folder, obj_dir=obj_dir,
                    texture=tex, mesh=mesh,
                    badge_frac=round(vis["frac"], 3),
                    badge_visible_t0=bool(vis["frac"] >= C.BADGE_READABLE_MIN),
                    badge_hidden_t0=bool(vis["frac"] <= C.BADGE_HIDDEN_MAX),
                    occlusion=float(occl.get(int(oid), 0.0)))
        out.append(site)
    return out


def _eligible_triples(cands, gt_lookup: Dict, rng,
                      wanted_largest_rank: Optional[int] = None):
    """Yield every triple satisfying the cheap, pre-bake constraints.

    Crucially, this is an iterator rather than a "first matching triple"
    selector. Revealability can only be evaluated by temporarily parking the
    non-target clutter. A first candidate can fail because one marked target
    occludes another even though a later candidate in the *same scene* is fully
    revealable. Skipping the entire scene in that case wastes valid scenes.
    """
    if wanted_largest_rank is not None and wanted_largest_rank not in (0, 1, 2):
        raise ValueError("wanted_largest_rank must be None, 0, 1, or 2; "
                         f"got {wanted_largest_rank}")

    # Deterministic but episode-varied order prevents object-list order from
    # deciding which physical assets receive labels.
    triples = list(itertools.combinations(cands, 3))
    rng.shuffle(triples)
    for raw in triples:
        vols = [_volume(s, gt_lookup) for s in raw]
        order = sorted(range(3), key=lambda i: vols[i])
        small, runner, big = (vols[order[0]], vols[order[1]], vols[order[2]])
        largest_runnerup_ratio = big / runner
        if not (C.TRIPLE_LARGEST_RUNNERUP_MIN_RATIO <= largest_runnerup_ratio
                <= C.TRIPLE_LARGEST_RUNNERUP_MAX_RATIO):
            continue
        # Manipulation must reveal a MARKER, not merely an arbitrary part of an
        # object. _eligible deliberately retains the most-visible viable site
        # per object, so passing this gate means no offered site on at least one
        # selected object is readable without moving clutter.
        if not any(s["badge_hidden_t0"] for s in raw):
            continue
        by_occ = sorted(raw, key=lambda s: (s["occlusion"], s["obj_id"]))
        largest = raw[order[2]]
        largest_rank = by_occ.index(largest)  # 0=least, 2=most occluded
        if (wanted_largest_rank is not None and
                largest_rank != wanted_largest_rank):
            continue
        yield raw, {
            "largest_obj_id": largest["obj_id"],
            "largest_rank": largest_rank,
            "small": small,
            "runner": runner,
            "big": big,
            "largest_runnerup_ratio": largest_runnerup_ratio,
        }


def _triple(cands, gt_lookup, rng, wanted_largest_rank: Optional[int] = None):
    """Backward-compatible first pre-bake candidate selector.

    Production callers should use :func:`find_revealable_triple`, which checks
    every eligible candidate against the clutter-parking oracle before deciding
    a scene has failed.
    """
    for raw, selection in _eligible_triples(cands, gt_lookup, rng,
                                             wanted_largest_rank):
        return _finalize_triple(raw, selection, gt_lookup, rng,
                                wanted_largest_rank)
    return None, None
    # NOTE: _triple() is a legacy first-match helper; production uses
    # find_revealable_triple(). Extremum is applied there (see below).


def _finalize_triple(raw, selection: dict, gt_lookup: Dict, rng,
                     wanted_largest_rank: Optional[int],
                     extremum: str = "largest"):
    """Copy, label, and write final GT metadata for an oracle-valid triple.

    ``extremum`` selects which physical extreme the QUESTION asks about:
    ``"largest"`` (default, original behaviour) or ``"smallest"``.  Only the
    requested answer letter changes — the physical scene, the occlusion tier
    (always defined by the LARGEST object's rank), and every candidate-gating
    ratio are unchanged.  This keeps ``gt_most_occluded`` etc. meaning exactly
    what it did before (the largest object's occlusion stratum), independent of
    whether the episode ends up asking for the largest or the smallest object.
    """
    if extremum not in ("largest", "smallest"):
        raise ValueError(f"extremum must be 'largest' or 'smallest'; got {extremum}")
    labelled = [dict(site) for site in raw]
    rng.shuffle(labelled)
    for site, label in zip(labelled, _LABELS):
        site["label"] = label
        site["volume_cm3"] = round(_volume(site, gt_lookup), 1)
    # Label→volume so we can name the largest AND smallest object's letters.
    vol_by_label = {s["label"]: s["volume_cm3"] for s in labelled}
    largest_label = max(vol_by_label, key=vol_by_label.get)
    smallest_label = min(vol_by_label, key=vol_by_label.get)
    answer_label = largest_label if extremum == "largest" else smallest_label
    largest_rank = selection["largest_rank"]
    return labelled, {
        "gt_letter": answer_label,
        # The task polarity for this episode and the two extreme letters, so
        # downstream (question text, review UI, analysis) is unambiguous.
        "comparison_extremum": extremum,
        "requested_extremum_answer": answer_label,
        "largest_letter": largest_label,
        "smallest_letter": smallest_label,
        # Tier is ALWAYS the largest physical object's occlusion rank (intact).
        "triple_tier": ("gt_least_occluded", "gt_middle_occluded",
                        "gt_most_occluded")[largest_rank],
        "largest_occlusion_rank": largest_rank,
        "tier_as_requested": (wanted_largest_rank is None or
                              largest_rank == wanted_largest_rank),
        "largest_runnerup_ratio": round(selection["largest_runnerup_ratio"], 2),
        "largest_smallest_ratio": round(selection["big"] / selection["small"], 2),
        "n_labels_hidden_t0": sum(s["badge_hidden_t0"] for s in labelled),
        "occlusions": {s["label"]: round(s["occlusion"], 2)
                       for s in labelled},
    }


def find_revealable_triple(demo, cands, gt_lookup: Dict, rng,
                          wanted_largest_rank: Optional[int] = None,
                          extremum: str = "largest"):
    """Select a triple only after checking its clutter-parking reveal oracle.

    A scene often contains multiple pre-bake triples. Earlier code selected the
    first cheap-match then rejected the *whole scene* if that triple had target-
    on-target occlusion. Here, every candidate is tested before declaring the
    scene unusable. This changes only scene yield, never the task definition:
    each returned triple still has a badge hidden initially and every badge
    readable after non-target clutter is parked.

    Returns ``(triple, gt, diagnostics)``; ``triple``/``gt`` are ``None`` if no
    candidate survives. No texture is baked while searching, so failed
    candidates do not mutate the rendered episode.
    """
    n_prebake = n_oracle_failed = 0
    for raw, selection in _eligible_triples(cands, gt_lookup, rng,
                                             wanted_largest_rank):
        n_prebake += 1
        reveal_bad = _verify_reveal_oracle(demo, raw)
        if reveal_bad:
            n_oracle_failed += 1
            continue
        triple, gt = _finalize_triple(raw, selection, gt_lookup, rng,
                                       wanted_largest_rank, extremum=extremum)
        return triple, gt, {"prebake_candidates": n_prebake,
                            "oracle_failed_candidates": n_oracle_failed}
    return None, None, {"prebake_candidates": n_prebake,
                         "oracle_failed_candidates": n_oracle_failed}


def _verify_reveal_oracle(demo, triple):
    """Ensure the hidden badge is revealable after non-target clutter is parked.

    Triple selection already requires at least one *badge* to be hidden in the
    initial scene. This checks the intended manipulation oracle: moving only
    non-target clutter exposes that badge. Other target badges must also remain
    readable, otherwise the scene could require destructive target removal.
    """
    import pybullet as p
    hidden_initially = [s for s in triple if s["badge_hidden_t0"]]
    keep = {s["obj_id"] for s in triple}
    stash = {}
    for oid in demo.env.obj_ids:
        if oid not in keep:
            stash[oid] = p.getBasePositionAndOrientation(oid)
            p.resetBasePositionAndOrientation(oid, [5, 5, 5], stash[oid][1])
    try:
        _, _, seg = demo.env.camera.get_cam_img_pybullet()
        name_by_id = dict(zip(demo.env.obj_ids, demo.env.obj_names))
        for site in triple:
            vis = ML.badge_visible_frac(demo.env.camera, seg, site["world_point"],
                                        site["badge_cm_actual"], site["obj_id"])
            site["badge_frac_revealed"] = round(vis["frac"], 3)
            site["revealed_blocker"] = name_by_id.get(vis["top_other"])
    finally:
        for oid, (pos, quat) in stash.items():
            p.resetBasePositionAndOrientation(oid, pos, quat)
    # All labels must be readable in the oracle, and in particular every
    # initially hidden badge must cross the reveal threshold. Returning the
    # concrete bad sites keeps the caller's rejection diagnostics audit-friendly.
    return [s for s in triple if s["badge_frac_revealed"] < C.BADGE_READABLE_MIN]


def build_marked_descriptor(triple: list, gt: dict) -> dict:
    """Build the complete marked descriptor shared by generation and replay.

    Keeping this separate from texture baking makes the answer-bearing metadata
    serialisable into a frozen scene package without recreating a scene or
    rerunning triple selection later.
    """
    q_desc = Q.build_question(triple, gt)
    q_desc.update({
        "target_name": triple[0]["name"],  # bookkeeping only
        "comparison_object": triple[1]["name"],
        "marker_obj_ids": {s["label"]: s["obj_id"] for s in triple},
        "marker_obj_names": {s["label"]: s["name"] for s in triple},
        "require_target_visible": False,
        # Metadata only: no runtime guard consumes this. Removing a labelled
        # target is intentionally an agent decision/error, not prevented.
        "protected_obj_ids": [s["obj_id"] for s in triple],
        "marked_sites": [
            {k: s[k] for k in ("label", "name", "folder", "obj_id",
                                "badge_cm_actual", "badge_frac", "occlusion",
                                "volume_cm3", "world_point", "glyph_deg",
                                "badge_texture", "badge_rendered_frac",
                                "badge_frac_revealed", "painted_texels",
                                "badge_paint_frac") if k in s}
            for s in triple],
    })
    return q_desc


def prepare_marked_episode(demo, folder_by_name: Dict[str, str],
                           asset_path: str, gt_lookup: Dict, rng, tex_dir: str,
                           variant: str = "shape", occ_stats: Optional[dict] = None,
                           wanted_largest_rank: Optional[int] = None,
                           extremum: str = "largest",
                           ) -> Tuple[Optional[dict], object]:
    """Select three objects, bake A/B/C, validate, and return a descriptor.

    ``extremum`` ("largest" | "smallest") selects the question polarity; only
    the answer letter changes.  The scene, badges and occlusion tier are
    identical either way (see :func:`_finalize_triple`).
    """
    if occ_stats is None:
        _, occ_stats = demo.pick_most_occluded_target()
    _, _, seg = demo.env.camera.get_cam_img_pybullet()
    cands = _eligible(demo, folder_by_name, asset_path, occ_stats, seg)
    if len(cands) < 3:
        return None, "marked_few_eligible_sites"

    triple, gt, select_diag = find_revealable_triple(
        demo, cands, gt_lookup, rng, wanted_largest_rank, extremum=extremum)
    if triple is None:
        # Preserve the existing outward reason while retaining a detailed
        # diagnostic in logs for scene-bank yield audits.
        if select_diag["prebake_candidates"]:
            print("  ⚠ no revealable marked triple: "
                  f"{select_diag['oracle_failed_candidates']}/"
                  f"{select_diag['prebake_candidates']} candidates failed "
                  "target-on-target visibility after clutter parking")
        return None, "marked_no_triple"

    os.makedirs(tex_dir, exist_ok=True)
    baked = []
    for site in triple:
        png = os.path.join(tex_dir,
            f"badge_{site['label']}_{ML.safe_name(site['folder'])[:40]}.png")
        try:
            ML.bake_badge(site["texture"], site, site["mesh"], site["label"],
                          png, variant=variant)
        except ValueError as exc:
            print(f"  ⚠ badge bake failed for {site['label']}: {exc}")
            return None, "marked_badge_bake_failed"
        ML.apply_texture(site["obj_id"], png)
        site["badge_texture"] = png
        baked.append(png)

    # Blender before/after texture differential. The initial warm-up avoids the
    # first-render table-texture change being mistaken for a badge delta.
    rgb_before = None
    try:
        demo.env.camera.get_cam_img()
        rgb_before, _, _ = demo.env.camera.get_cam_img()
        rgb_before = np.asarray(rgb_before[:, :, :3], dtype=np.int16).copy()
    except Exception as exc:
        print(f"  ⚠ pre-badge render failed: {exc}")
    for site in triple:
        try:
            demo.env.camera.set_object_texture(site["obj_id"], site["badge_texture"])
        except Exception as exc:
            print(f"  ⚠ Blender badge push failed for {site['label']}: {exc}")
    try:
        rgb_after, _, _ = demo.env.camera.get_cam_img()
        rgb_after = np.asarray(rgb_after[:, :, :3], dtype=np.int16)
        for site in triple:
            frac = (ML.badge_delta_frac(rgb_before, rgb_after, demo.env.camera,
                                        site["world_point"], site["badge_cm_actual"])
                    if rgb_before is not None else 0.0)
            site["badge_rendered_frac"] = round(frac, 3)
        # Small/far badges can produce a sub-threshold *image* delta even when
        # the texture bake is sound. The post-bake alpha-retention assertion is
        # the hard admission check; preserve this Blender differential as an
        # audit warning rather than throwing away an otherwise valid triple.
        bad = [s for s in triple if s["badge_visible_t0"] and
               s["badge_rendered_frac"] < C.BADGE_RENDERED_MIN]
        if bad:
            print("  ⚠ low Blender badge delta (audit): " + ", ".join(
                f"{s['label']}={s['badge_rendered_frac']:.2f}" for s in bad))
    except Exception as exc:
        # A renderer outage should not silently claim validation succeeded; the
        # PyBullet bake assertion has passed, but callers can audit this state.
        print(f"  ⚠ badge render check skipped: {exc}")
        for site in triple:
            site.setdefault("badge_rendered_frac", None)

    reveal_bad = _verify_reveal_oracle(demo, triple)
    if reveal_bad:
        print("  ⚠ reveal oracle failed: " + ", ".join(
            f"{s['label']}={s['badge_frac_revealed']:.2f}" for s in reveal_bad))
        return None, "marked_reveal_incomplete"

    q_desc = build_marked_descriptor(triple, gt)
    info = {"baked": baked, "gt": gt, "n_eligible": len(cands),
            "selection": select_diag,
            "occlusions": {s["label"]: round(s["occlusion"], 2) for s in triple}}
    return q_desc, info
