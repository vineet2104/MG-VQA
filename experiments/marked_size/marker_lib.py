"""Bake a physical label badge into an object's texture map.

Pipeline (all offline geometry — no Blender needed):

    1. settled pose (pos, quat) -> rotation R
    2. mesh faces + UVs from the asset OBJ (trimesh)
    3. keep faces whose WORLD normal is within UP_ANGLE_DEG of +Z
    4. grow the largest contiguous up-facing patch, ranked by projected area
    5. badge centre = that patch's UV centroid
    6. badge radius in UV = f(desired cm, local texel density)   <- cm, not UV units
    7. paint the badge ONLY inside the patch's rasterised UV mask  <- no island bleed
    8. verify visibility with a camera ray test

The badge lives in the texture, so it is glued to the surface: it translates
and rotates with the object exactly like paint, and it shows up in BOTH
renderers (PyBullet loads the MTL texture, Blender imports it).

Nothing here mutates the shared asset tree — baked PNGs go to a scratch dir
and are applied at runtime via p.changeVisualShape(textureUniqueId=...).
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

from . import config as C

_MESH_CACHE: Dict[str, object] = {}

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


# ---------------------------------------------------------------------------
# Mesh + UV helpers
# ---------------------------------------------------------------------------

def load_mesh(obj_dir: str):
    """Load the visual mesh with UVs (cached).  Returns None if unusable."""
    if obj_dir in _MESH_CACHE:
        return _MESH_CACHE[obj_dir]
    import trimesh
    path = os.path.join(obj_dir, "textured_simplified.obj")
    if not os.path.isfile(path):
        cands = [f for f in os.listdir(obj_dir)
                 if f.endswith(".obj") and "vhacd" not in f]
        if not cands:
            _MESH_CACHE[obj_dir] = None
            return None
        path = os.path.join(obj_dir, cands[0])
    try:
        m = trimesh.load(path, process=False, force="mesh")
    except Exception:
        m = None
    if m is not None:
        uv = getattr(m.visual, "uv", None)
        if uv is None or len(uv) != len(m.vertices):
            m = None          # no usable per-vertex UVs
    _MESH_CACHE[obj_dir] = m
    return m


def texture_path(obj_dir: str) -> Optional[str]:
    for name in ("texture_map.png", "texture.png", "material_0.png"):
        cand = os.path.join(obj_dir, name)
        if os.path.isfile(cand):
            return cand
    return None


def _quat_to_R(quat_xyzw) -> np.ndarray:
    return np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)


def pick_marker_site(
    mesh,
    body_pos,
    body_quat,
    visual_offset: Optional[Tuple] = None,
    up_angle_deg: float = None,
    min_patch_area_cm2: float = None,
    tex_size: int = None,
    validate=None,
    cam_eye=None,
    max_sites: int = 1,
    debug: bool = False,
):
    """Choose an up-facing surface patch to host the badge.

    Returns dict with keys:
        faces        - face indices forming the chosen patch
        uv_center    - (u, v) badge centre in texture space, 0-1
        uv_radius    - badge radius in UV units for BADGE_DIAMETER_CM
        world_point  - badge centre in world coords (for the ray test)
        patch_cm2    - projected area of the patch
        up_cos       - alignment of the patch normal with +Z
    or None when the object has no suitable patch.
    """
    """Returns a LIST of valid badge sites (best patch first), possibly empty."""
    up_angle_deg = C.UP_ANGLE_DEG if up_angle_deg is None else up_angle_deg
    min_patch_area_cm2 = (C.MIN_PATCH_AREA_CM2 if min_patch_area_cm2 is None
                          else min_patch_area_cm2)
    tex_size = C.BAKE_MAX_TEX if tex_size is None else tex_size

    R = _quat_to_R(body_quat)
    # URDF <visual><origin> offset: mesh coords are expressed in the visual
    # frame, which may be translated/rotated w.r.t. the body frame.
    if visual_offset:
        v_pos, v_orn = visual_offset
        R_v = _quat_to_R(v_orn)
        R_total = R @ R_v
        t_total = np.asarray(body_pos, float) + R @ np.asarray(v_pos, float)
    else:
        R_total = R
        t_total = np.asarray(body_pos, float)

    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)
    UV = np.asarray(mesh.visual.uv)

    # World-space face normals
    fn = np.asarray(mesh.face_normals)
    fn_world = fn @ R_total.T
    # abs(): face winding is inconsistent across these assets, so a genuinely
    # up-facing face can compute as down-facing.  Accept both signs for recall
    # and let the ray-test validator reject sites that are actually underneath.
    up_cos = np.abs(fn_world[:, 2])
    cand = np.where(up_cos > np.cos(np.radians(up_angle_deg)))[0]
    if debug:
        print(f"    [dbg] faces={len(faces)} up-facing={len(cand)} "
              f"(thresh {up_angle_deg}deg)")
    if len(cand) == 0:
        return []

    # Projected (top-down) area of each candidate face, in cm^2
    tri = verts[faces[cand]] @ R_total.T + t_total
    ab = tri[:, 1] - tri[:, 0]
    ac = tri[:, 2] - tri[:, 0]
    cross = np.cross(ab, ac)
    proj_area_cm2 = np.abs(cross[:, 2]) * 0.5 * 1e4     # m^2 -> cm^2

    # Group candidate faces into CONTIGUOUS patches via mesh face adjacency.
    #
    # An earlier version binned faces by quantised normal + 2 cm height bands.
    # That fragmented curved surfaces into many sub-threshold pieces, so almost
    # nothing passed MIN_PATCH_AREA_CM2 -- only ~4 of 15 objects per scene were
    # eligible, even though a pose-free analysis shows 89% of assets have UV
    # room for a 2 cm badge (median 10.8 cm).  Connected components over the
    # up-facing subgraph give real surface patches instead.
    cz = tri[:, :, 2].mean(axis=1)
    pos_in_cand = {int(f): i for i, f in enumerate(cand)}
    groups: Dict[tuple, List[int]] = {}
    try:
        import trimesh.graph as _tg
        adj = np.asarray(mesh.face_adjacency)
        up_set = set(int(f) for f in cand)
        edges = [(int(a), int(b)) for a, b in adj
                 if int(a) in up_set and int(b) in up_set]
        comps = _tg.connected_components(edges, nodes=list(up_set))
        for ci, comp in enumerate(comps):
            idxs = [pos_in_cand[int(f)] for f in comp if int(f) in pos_in_cand]
            if idxs:
                groups[(ci,)] = idxs
    except Exception:
        # Fallback: coarse normal/height binning (previous behaviour).
        key_n = np.round(fn_world[cand] * 2).astype(int)
        key_z = np.round(cz * 25).astype(int)
        for i, k in enumerate(tuple(kn) + (kz,)
                              for kn, kz in zip(key_n, key_z)):
            groups.setdefault(k, []).append(i)

    ranked = []
    for k, idxs in groups.items():
        area = float(proj_area_cm2[idxs].sum())
        if area < min_patch_area_cm2:
            continue
        # prefer big, high, strongly up-facing patches
        # Prefer HIGH patches.  abs(normal.z) accepts undersides too (needed
        # because face winding is unreliable), so top and bottom faces score
        # almost identically -- and the underside then burns a candidate slot
        # before the ray test rejects it.  Weighting height heavily puts the
        # real top face first.  (alphabet block: top face was tried second and
        # never reached, giving 0 sites on an object with a 4x4cm flat top.)
        z_rel = cz[idxs].mean()
        score = area * (1.0 + up_cos[cand[idxs]].mean()) * (1.0 + 8.0 * z_rel)
        ranked.append((score, idxs, area))
    if debug:
        areas = sorted((float(proj_area_cm2[i].sum()) for _, i, _ in
                        [(0, v, 0) for v in groups.values()]), reverse=True)[:5]
        print(f"    [dbg] contiguous patches={len(groups)} "
              f"top projected areas cm2={[round(a,1) for a in areas]} "
              f"(need >= {min_patch_area_cm2})")
    if not ranked:
        return []
    ranked.sort(key=lambda t: -t[0])

    # Try patches best-first; a patch is only usable if its UV island has room
    # for the badge (a patch can span DISCONNECTED islands, in which case the
    # naive UV centroid lands in empty texture space and gets clipped away).
    # Collect up to ``max_sites`` valid ones so the CALLER can choose by
    # in-scene visibility: taking only the biggest/highest patch systematically
    # lands the badge where nothing can cover it, which starves the benchmark of
    # "must be revealed" episodes.
    found = []
    for _, idxs, patch_cm2 in ranked[:12]:
        patch_faces = cand[idxs]
        uv_tri = UV[faces[patch_faces]]                       # (n,3,2)
        world_tri = verts[faces[patch_faces]] @ R_total.T + t_total

        # cm -> UV conversion via this patch's texel density
        a = world_tri[:, 1] - world_tri[:, 0]
        b = world_tri[:, 2] - world_tri[:, 0]
        world_cm2 = float((0.5 * np.linalg.norm(np.cross(a, b), axis=1)).sum() * 1e4)
        ua = uv_tri[:, 1] - uv_tri[:, 0]
        ub = uv_tri[:, 2] - uv_tri[:, 0]
        uv_area = float((0.5 * np.abs(ua[:, 0] * ub[:, 1] - ua[:, 1] * ub[:, 0])).sum())
        if world_cm2 <= 1e-9 or uv_area <= 1e-12:
            continue
        uv_per_cm = float(np.sqrt(uv_area / world_cm2))
        want_r = 0.5 * C.BADGE_DIAMETER_CM * uv_per_cm         # in UV units

        # Most-interior point of the rasterised island, via distance transform.
        mask = np.asarray(_patch_uv_mask_from(uv_tri, tex_size))
        import cv2
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
        max_r_uv = float(dist[py, px]) / float(tex_size)       # px -> UV units
        # Enforce a hard floor in CENTIMETRES rather than shrinking the badge
        # below legibility on a small island.
        min_r = 0.5 * C.BADGE_MIN_CM * uv_per_cm
        if debug:
            print(f"    [dbg] patch {patch_cm2:6.1f}cm2  uv/cm={uv_per_cm:.4f}  "
                  f"island_max_badge={2*max_r_uv/max(uv_per_cm,1e-9):5.2f}cm  "
                  f"need>={C.BADGE_MIN_CM}cm  "
                  f"{'OK' if max_r_uv >= min_r else 'ISLAND TOO SMALL'}")
        if max_r_uv < min_r:
            continue                                           # island too small
        uv_radius = min(want_r, max_r_uv)
        uv_center = np.array([(px + 0.5) / tex_size,
                              1.0 - (py + 0.5) / tex_size])

        # Exact 3D surface point: find the UV triangle containing uv_center and
        # reuse its barycentric coords on the world triangle.
        hit = _uv_to_world_and_basis(uv_center, uv_tri, world_tri)
        if hit is None:
            continue
        world_point, glyph_deg, mirror = hit

        # SELF-occlusion only: would the badge be visible if this object were
        # ALONE in the scene?  Occlusion by *other* objects must NOT disqualify
        # a site -- that is precisely the episode we want (the model has to
        # manipulate to reveal the label).  So this casts against this object's
        # own mesh only.
        if cam_eye is not None and not _self_visible(
                mesh, R_total, t_total, cam_eye, world_point):
            if debug:
                print("    [dbg]   -> rejected: self-occluded")
            continue

        # Remaining caller-supplied checks (e.g. inside the camera frustum).
        if validate is not None and not validate(world_point):
            if debug:
                print("    [dbg]   -> rejected: validate() (out of frame)")
            continue

        found.append({
            "glyph_deg": float(glyph_deg),
            "glyph_mirror": bool(mirror),
            "faces": patch_faces,
            "uv_center": (float(uv_center[0]), float(uv_center[1])),
            "uv_radius": float(uv_radius),
            "uv_radius_wanted": float(want_r),
            "world_point": [float(x) for x in world_point],
            "patch_cm2": float(patch_cm2),
            "up_cos": float(up_cos[patch_faces].mean()),
            "uv_per_cm": uv_per_cm,
            "badge_cm_actual": float(2.0 * uv_radius / max(uv_per_cm, 1e-9)),
        })
        if len(found) >= max_sites:
            break
    return found


def _uv_to_world_and_basis(uv_pt, uv_tri, world_tri):
    """UV point -> (3D surface point, glyph rotation deg, mirror flag).

    The glyph must be drawn axis-aligned in *texture* space, but UV islands
    carry no relation to world orientation, so an unrotated letter lands at an
    arbitrary angle on the object (a 180-degree "B" reads as "8").

    Using the containing triangle's affine UV->world Jacobian M:
      * texture +u maps to world M[:,0], texture +v maps to world M[:,1]
      * the camera is top-down with up=[0,1,0], so image-right = world +X and
        image-up = world +Y
      * solve for the sprite rotation whose "up" maps to world +Y
      * if the island is mirrored, the letter reads backwards -> flip flag
    """
    p0, p1, p2 = uv_tri[:, 0], uv_tri[:, 1], uv_tri[:, 2]
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = uv_pt[None, :] - p0
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-14
    if not ok.any():
        return None
    u = np.full(len(den), -1.0)
    v = np.full(len(den), -1.0)
    u[ok] = (v2[ok, 0] * v1[ok, 1] - v1[ok, 0] * v2[ok, 1]) / den[ok]
    v[ok] = (v0[ok, 0] * v2[ok, 1] - v2[ok, 0] * v0[ok, 1]) / den[ok]
    inside = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
    idx = int(np.argmax(inside)) if inside.any() else None
    if idx is None:
        return None
    w0, w1, w2 = world_tri[idx, 0], world_tri[idx, 1], world_tri[idx, 2]
    world_point = w0 + u[idx] * (w1 - w0) + v[idx] * (w2 - w0)

    # Affine UV->world Jacobian for this triangle: M = [w1-w0, w2-w0] . inv(UV)
    duv = np.array([[v0[idx, 0], v1[idx, 0]],
                    [v0[idx, 1], v1[idx, 1]]])
    try:
        inv = np.linalg.inv(duv)
    except np.linalg.LinAlgError:
        return world_point, 0.0, False
    dW = np.stack([w1 - w0, w2 - w0], axis=1)      # 3x2
    M = dW @ inv                                    # world per unit (u, v)
    u_w, v_w = M[:, 0], M[:, 1]

    # Sprite drawn upright has its up along +v.  Rotating the sprite CCW by
    # theta (texture pixel space, y down) sends its up to (du, dv) =
    # (-sin, +cos), whose world image-plane x-component must vanish:
    #     -sin(t) * u_w.x + cos(t) * v_w.x = 0  ->  t = atan2(v_w.x, u_w.x)
    t = np.arctan2(v_w[0], u_w[0])
    up_y = -np.sin(t) * u_w[1] + np.cos(t) * v_w[1]
    if up_y < 0:                                    # pointing down -> flip 180
        t += np.pi
    # Sprite right maps to cos(t)*u_w + sin(t)*v_w; if that points image-left
    # the island is mirrored and the glyph would read backwards.
    right_x = np.cos(t) * u_w[0] + np.sin(t) * v_w[0]
    mirror = bool(right_x < 0)
    return world_point, float(np.degrees(t)), mirror


def _self_visible(mesh, R_total, t_total, cam_eye, world_point,
                  tol: float = None) -> bool:
    """Would the badge be visible if this object were ALONE in the scene?

    Casts a camera ray against the object's own mesh in its local frame and
    checks that the badge point is the FIRST surface along that ray.  This is
    the right notion of a valid site: it approximates what the model sees once
    manipulation has cleared the surrounding clutter.
    """
    tol = C.SELF_RAY_TOL_M if tol is None else tol
    Rt = np.asarray(R_total).T
    o = Rt @ (np.asarray(cam_eye, float) - np.asarray(t_total, float))
    q = Rt @ (np.asarray(world_point, float) - np.asarray(t_total, float))
    d = q - o
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return False
    d = d / L
    try:
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=o[None, :], ray_directions=d[None, :],
            multiple_hits=True)
    except Exception:
        return True          # no ray backend -> do not over-filter
    if len(locs) == 0:
        return False
    dmin = float(np.linalg.norm(np.asarray(locs) - o[None, :], axis=1).min())
    return dmin >= L - tol


def safe_name(s: str) -> str:
    """Filesystem-safe token (asset folders contain '/', which would other-
    wise silently create nested directories when joined into a filename)."""
    return s.replace(os.sep, "__").replace("/", "__")


def _patch_uv_mask_from(uv_tri, size: int) -> Image.Image:
    """Rasterise UV triangles (n,3,2) into a mask of the given size."""
    mask = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(mask)
    for pts in uv_tri:
        xy = [(float(u) * (size - 1), (1.0 - float(v)) * (size - 1))
              for u, v in pts]
        dr.polygon(xy, fill=255)
    return mask


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------

def _patch_uv_mask(mesh, patch_faces, size: int) -> Image.Image:
    """Rasterise the patch's UV triangles into a mask image.

    The badge is clipped to this mask so paint cannot bleed across a UV
    island boundary and reappear elsewhere on the object.
    """
    UV = np.asarray(mesh.visual.uv)
    faces = np.asarray(mesh.faces)
    return _patch_uv_mask_from(UV[faces[np.asarray(patch_faces)]], size)


def _font(px: int):
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, px)
            except Exception:
                pass
    return ImageFont.load_default()


def bake_badge(
    src_texture: str,
    site: dict,
    mesh,
    label: str,
    out_path: str,
    variant: str = "circle",
) -> str:
    """Composite a badge into a copy of the texture; return the new path."""
    img = Image.open(src_texture).convert("RGB")
    if max(img.size) > C.BAKE_MAX_TEX:
        img = img.resize((C.BAKE_MAX_TEX, C.BAKE_MAX_TEX), Image.LANCZOS)
    size = img.size[0]

    u, v = site["uv_center"]
    r = max(3.0, site["uv_radius"] * size)
    cx = float(u) * (size - 1)
    cy = (1.0 - float(v)) * (size - 1)

    # ── Build the badge as an RGBA sprite, then rotate it so the glyph is
    #    upright in the RENDER (UV space carries no world orientation).
    pad = int(r * 0.35) + 4
    S = int(2 * (r + pad))
    sprite = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sprite)
    box = [pad, pad, S - pad, S - pad]
    ring = max(1, int(r * C.RING_FRAC))

    shape = C.BADGE_SHAPES.get(label, "circle") if variant == "shape" else "circle"
    txt_y = S / 2
    txt_frac = C.TEXT_FRAC
    if shape == "square":
        ds.rectangle(box, fill=C.BADGE_FILL + (255,),
                     outline=C.BADGE_RING + (255,), width=ring)
    elif shape == "triangle":
        # Equilateral triangle inscribed in the badge circle, apex up.  Its
        # usable interior is much smaller than a disc, so the glyph is shrunk
        # and nudged down toward the centroid to stay inside the outline.
        #
        # NOTE: use sprite-LOCAL names here.  This branch previously assigned
        # `cx, cy = S/2, S/2`, clobbering the texture-space paste centre
        # computed from uv_center above -- so triangle badges were pasted at the
        # texture's top-left corner, landed outside the UV island, and were
        # clipped to zero.  Circles and squares never touched cx/cy, which is
        # why only label C ever failed.
        sx = sy = S / 2.0
        pts = [(sx, sy - r),
               (sx - 0.866 * r, sy + 0.5 * r),
               (sx + 0.866 * r, sy + 0.5 * r)]
        ds.polygon(pts, fill=C.BADGE_FILL + (255,))
        ds.line(pts + [pts[0]], fill=C.BADGE_RING + (255,), width=ring)
        txt_y = sy + 0.14 * r
        txt_frac = C.TEXT_FRAC * 0.72
    else:
        ds.ellipse(box, fill=C.BADGE_FILL + (255,),
                   outline=C.BADGE_RING + (255,), width=ring)
    f = _font(max(6, int(r * txt_frac)))
    ds.text((S / 2, txt_y), label, fill=C.BADGE_TEXT + (255,),
            anchor="mm", font=f)

    if site.get("glyph_mirror"):
        sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)
    sprite = sprite.rotate(site.get("glyph_deg", 0.0), resample=Image.BICUBIC,
                           expand=True)

    # Composite the sprite, clipped to the patch's UV island (no bleed).
    layer = Image.new("RGB", img.size, (0, 0, 0))
    lmask = Image.new("L", img.size, 0)
    ox, oy = int(cx - sprite.size[0] / 2), int(cy - sprite.size[1] / 2)
    alpha = sprite.split()[-1]
    layer.paste(sprite.convert("RGB"), (ox, oy), alpha)
    lmask.paste(alpha, (ox, oy))

    patch_mask = _patch_uv_mask(mesh, site["faces"], size)
    clipped = Image.fromarray(
        (np.asarray(lmask) * (np.asarray(patch_mask) > 0)).astype(np.uint8))
    img.paste(layer, (0, 0), clipped)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    painted = int((np.asarray(clipped) > 0).sum())
    sprite_texels = int((np.asarray(alpha) > 0).sum())
    paint_frac = float(painted) / max(sprite_texels, 1)
    site["painted_texels"] = painted
    site["badge_paint_frac"] = paint_frac
    site["badge_px_in_tex"] = float(r)
    if paint_frac < C.BADGE_PAINT_MIN:
        raise ValueError(
            f"Badge {label} survived UV clipping at only {paint_frac:.1%} "
            f"({painted}/{sprite_texels} texels; need {C.BADGE_PAINT_MIN:.0%})")
    return out_path


def apply_texture(body_id: int, png_path: str, link_index: int = -1) -> int:
    """Hot-swap a body's texture (UVs preserved, pose untouched)."""
    tid = p.loadTexture(png_path)
    p.changeVisualShape(body_id, link_index, textureUniqueId=tid)
    return tid


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def marker_visible(cam_eye, world_point, body_id: int, eps: float = 0.002,
                   tol: float = None) -> dict:
    """Ray-test from the camera to the badge centre.

    Visible iff the first hit is this body AND the hit lands at the badge
    point.  Checking the body id alone gives false positives: a site on the
    object's underside is occluded by the object's own top surface, which is
    the *same* body, so an id-only test wrongly reports it as visible.
    """
    tol = C.RAY_HIT_TOL_M if tol is None else tol
    src = np.asarray(cam_eye, float)
    dst = np.asarray(world_point, float)
    d = dst - src
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return {"visible": False, "hit_body": -1, "hit_err": 9.99}
    # Aim slightly past the point so a grazing surface still registers.
    hits = p.rayTest(src.tolist(), (src + d * (1.0 + eps / n)).tolist())
    if not hits:
        return {"visible": False, "hit_body": -1, "hit_err": 9.99}
    hit_body = int(hits[0][0])
    hit_pos = np.asarray(hits[0][3], float)
    err = float(np.linalg.norm(hit_pos - dst))
    return {"visible": bool(hit_body == body_id and err <= tol),
            "hit_body": hit_body, "hit_err": err}


def badge_visible_frac(camera, seg, world_point, badge_diam_cm: float,
                       body_id: int, shrink: float = 0.8) -> dict:
    """Fraction of the badge's on-screen disc that shows this object.

    This replaces a ray test, which was WRONG here: PyBullet's rayTest hits the
    *collision* mesh (textured_simplified_vhacd.obj -- a convex decomposition
    that bulges outward and seals concavities), while the badge lives on the
    *visual* mesh.  On a shoe interior or a bowl the hull is hit tens of mm
    early, so plainly visible badges were reported as hidden.

    The segmentation buffer is rendered from the VISUAL mesh, so it is the
    ground truth for what the VLM sees -- and it yields a graded measure
    instead of a single boolean.
    """
    row, col = project_to_pixel(camera, world_point)
    H, W = seg.shape[:2]
    if not (0 <= row < H and 0 <= col < W):
        return {"frac": 0.0, "row": row, "col": col, "radius_px": 0.0,
                "in_frame": False, "top_other": None}

    # On-screen badge radius: project a point offset by the badge radius along
    # world +X (perpendicular to this top-down view) and measure the shift.
    off = np.asarray(world_point, float) + np.array(
        [badge_diam_cm / 200.0, 0.0, 0.0])          # cm -> m, radius = diam/2
    r2, c2 = project_to_pixel(camera, off.tolist())
    radius_px = float(np.hypot(r2 - row, c2 - col))
    rr = max(2.0, radius_px * shrink)

    y0, y1 = int(max(0, row - rr)), int(min(H, row + rr + 1))
    x0, x1 = int(max(0, col - rr)), int(min(W, col + rr + 1))
    ys, xs = np.mgrid[y0:y1, x0:x1]
    disc = ((ys - row) ** 2 + (xs - col) ** 2) <= rr * rr
    if not disc.any():
        return {"frac": 0.0, "row": row, "col": col, "radius_px": radius_px,
                "in_frame": True, "top_other": None}
    patch = seg[y0:y1, x0:x1][disc]
    frac = float((patch == body_id).mean())

    other = patch[patch != body_id]
    top_other = None
    if other.size:
        vals, counts = np.unique(other, return_counts=True)
        top_other = int(vals[int(np.argmax(counts))])
    return {"frac": frac, "row": row, "col": col, "radius_px": radius_px,
            "in_frame": True, "top_other": top_other}


def project_to_pixel(camera, world_point) -> Tuple[int, int]:
    """Project a world point to (row, col) using the camera matrices."""
    view = np.asarray(camera.view_matrix, float).reshape(4, 4, order="F")
    proj = np.asarray(camera.projection_matrix, float).reshape(4, 4, order="F")
    pt = np.array([*world_point, 1.0])
    clip = proj @ (view @ pt)
    if abs(clip[3]) < 1e-9:
        return -1, -1
    ndc = clip[:3] / clip[3]
    col = int(round((ndc[0] * 0.5 + 0.5) * (camera.width - 1)))
    row = int(round((1.0 - (ndc[1] * 0.5 + 0.5)) * (camera.height - 1)))
    return row, col


def badge_rendered_frac(rgb, camera, world_point, badge_diam_cm: float,
                        shrink: float = 0.9) -> dict:
    """Fraction of the badge disc in a RENDERED RGB image that looks like badge.

    The seg-buffer check (badge_visible_frac) only proves the OBJECT is visible
    at that pixel -- not that the badge was actually painted there.  A frame
    mismatch between the site computation and the renderer put badges on faces
    the camera never sees, and nothing caught it.  Badges are achromatic
    (white fill, black ring/glyph), so "near-grey and bright-or-very-dark" is a
    good detector against these colourful, wood-backed scenes.
    """
    row, col = project_to_pixel(camera, world_point)
    H, W = rgb.shape[:2]
    if not (0 <= row < H and 0 <= col < W):
        return {"frac": 0.0, "row": row, "col": col}
    off = np.asarray(world_point, float) + np.array(
        [badge_diam_cm / 200.0, 0.0, 0.0])
    r2, c2 = project_to_pixel(camera, off.tolist())
    rr = max(2.0, float(np.hypot(r2 - row, c2 - col)) * shrink)
    y0, y1 = int(max(0, row - rr)), int(min(H, row + rr + 1))
    x0, x1 = int(max(0, col - rr)), int(min(W, col + rr + 1))
    ys, xs = np.mgrid[y0:y1, x0:x1]
    disc = ((ys - row) ** 2 + (xs - col) ** 2) <= rr * rr
    if not disc.any():
        return {"frac": 0.0, "row": row, "col": col}
    px = np.asarray(rgb)[y0:y1, x0:x1][disc].astype(np.int16)
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    achromatic = (mx - mn) <= 40
    bright = mx >= 170
    dark = mx <= 70
    frac = float((achromatic & (bright | dark)).mean())
    return {"frac": frac, "row": row, "col": col, "radius_px": rr}


def badge_delta_frac(rgb_before, rgb_after, camera, world_point,
                     badge_diam_cm: float, shrink: float = 0.9,
                     thresh: int = 25) -> float:
    """Fraction of the badge disc whose pixels CHANGED after the texture swap.

    Exact by construction: the only thing altered between the two renders is
    the badge, so a changed pixel means badge paint actually reached the image.
    Replaces a colour heuristic that false-positived on white/black packaging.
    """
    row, col = project_to_pixel(camera, world_point)
    H, W = rgb_after.shape[:2]
    if not (0 <= row < H and 0 <= col < W):
        return 0.0
    off = np.asarray(world_point, float) + np.array(
        [badge_diam_cm / 200.0, 0.0, 0.0])
    r2, c2 = project_to_pixel(camera, off.tolist())
    rr = max(2.0, float(np.hypot(r2 - row, c2 - col)) * shrink)
    y0, y1 = int(max(0, row - rr)), int(min(H, row + rr + 1))
    x0, x1 = int(max(0, col - rr)), int(min(W, col + rr + 1))
    ys, xs = np.mgrid[y0:y1, x0:x1]
    disc = ((ys - row) ** 2 + (xs - col) ** 2) <= rr * rr
    if not disc.any():
        return 0.0
    d = np.abs(rgb_after[y0:y1, x0:x1] - rgb_before[y0:y1, x0:x1]).sum(axis=2)
    return float((d[disc] > thresh).mean())
