"""Badge design knobs for the marked-size question type prototype.

Everything here is a tunable; nothing in this directory is imported by the
benchmark. Iterate freely.
"""

# ── Badge geometry ────────────────────────────────────────────────────
# 12.6 px/cm at the benchmark camera (1024 px over 0.812 m).
#   2.4 cm -> ~30 px, but the floor of 2.0 cm disqualified many assets: the
#   binding constraint is the largest inscribed circle in a single UV ISLAND on
#   an up-facing patch, not the object's physical area (a 4x4 cm alphabet block
#   maxed out at 1.51 cm, a 9.8 cm flower pot at 1.57 cm).  Identification is
#   not the skill under test, so we trade badge size for eligibility.
BADGE_DIAMETER_CM = 2.0      # ~25 px physical size on the object's surface
UP_ANGLE_DEG = 30.0          # face counts as "up-facing" if normal within this of +Z
MIN_PATCH_AREA_CM2 = 2.0     # up-facing patch must be at least this big to host a badge
# (30deg, not 50: at 50deg a steeply tilted surface still qualifies and
#  foreshortening crushed a 1.9cm badge to 37 rendered pixels)
# Hard floor on the achievable badge size: if the UV island cannot host at
# least this diameter, the site is rejected rather than shrinking the badge
# below legibility.  Lowered from 2.0 to admit assets whose UV islands are
# finely subdivided (flower pot 1.57 cm, alphabet block 1.51 cm both missed 1.6).
BADGE_MIN_CM = 1.5

# ── Badge appearance (achromatic on purpose: colour would confound a
#    size comparison, and would be harmful if reused for the colour type) ──
BADGE_FILL = (255, 255, 255)
BADGE_RING = (0, 0, 0)
BADGE_TEXT = (0, 0, 0)
RING_FRAC = 0.10             # ring thickness as fraction of badge radius
TEXT_FRAC = 1.25             # glyph height as fraction of badge radius

# ── Variants rendered for comparison ──────────────────────────────────
# "circle"  : both badges are discs (A and B differ only by the letter)
# "shape"   : A = disc, B = square  -> identity survives at low resolution via
#             the outline, the letter confirms it when resolution allows
VARIANTS = ("circle", "shape")
# Shape-coded identity, so a badge stays distinguishable at ~30 px even when
# the letter itself blurs.  A third target needs a third shape: without a "C"
# entry it silently fell back to a circle, making A and C indistinguishable.
BADGE_SHAPES = {"A": "circle", "B": "square", "C": "triangle"}

# Reject a candidate site unless the badge centre is genuinely visible:
# the camera ray must hit the target body *at that point*, not merely
# somewhere on the same body (which happens when the site is underneath).
RAY_HIT_TOL_M = 0.010

# Self-occlusion tolerance (metres) for the isolated-object ray cast.
SELF_RAY_TOL_M = 0.003

# Baked textures are large (~2.4 MB each).  The badge is fully determined by
# the site metadata, so pixels are regenerable and deleted by default.
KEEP_TEX = False

# ── Texture output ────────────────────────────────────────────────────
# Downscale the 4096² source maps when baking: far faster to write, and a
# 1.6 cm badge still gets plenty of texels.
BAKE_MAX_TEX = 2048

# Three visual identities for the three-way marked-size task.  The letter and
# shape redundantly identify the same object after it is revealed.
LABELS = ("A", "B", "C")

# Badge centre must be this far inside the image border (px) so the whole
# badge is in frame.  A ray test alone does not know about the camera FOV.
FRAME_MARGIN_PX = 45

# ── Pair selection (uses the benchmark's own occlusion machinery) ──────
OCC_HIGH = 0.50      # "hidden" object: occlusion fraction at or above this
OCC_LOW = 0.25       # "visible" object: occlusion fraction at or below this
# Volume-ratio BAND.  relative_size only enforces >= 1.1 with no upper bound,
# which lets through visually trivial pairs (we measured a median 8.0x and a
# max 248x, with only 2/32 episodes below 1.5x).  Bounding the band keeps the
# comparison genuinely discriminative.
MIN_VOL_RATIO = 1.20
MAX_VOL_RATIO = 3.00
# Three-way selection is driven by the ACTUAL manipulation requirement: at
# least one selected badge must be unreadable until clutter is moved.  We do
# not reject a triple because of an occlusion-rank target; rank is recorded and
# balanced later in the scene bank.  Keep the largest distinct from runner-up
# but not absurdly larger than it.  The third object may be small: the task is
# still to identify its painted label, not merely classify objects by size.
TRIPLE_LARGEST_RUNNERUP_MIN_RATIO = 1.20
TRIPLE_LARGEST_RUNNERUP_MAX_RATIO = 3.00

# Anti-shortcut pairing.  Big objects do not get buried; small ones do, so
# occlusion leaks the answer: "always pick the LESS-OCCLUDED object" scored
# 78.1% (vs the VLM's 71.9%).  Demanding the occluded object be the LARGER one
# fought that size distribution and was satisfiable in only 1/8 episodes.
#
# Instead, require BOTH targets to be similarly occluded: then "pick the
# visible one" carries no information because neither is clearly visible, and
# both labels have to be uncovered.
# Requiring BOTH targets to be similarly occluded turned out to be very rare
# (marked_no_pair 6x per success), so instead we BALANCE the cue directly:
#   even episode -> GT (the larger object) must be the MORE-occluded one
#   odd  episode -> GT must be the LESS-occluded one
# Over a run that puts "pick the less-occluded object" at exactly 50% (chance).
# The buried-is-larger case gets a wider ratio band because buried objects are
# small, so a *much* larger partner is what is actually findable.
OCC_BOTH_MIN = 0.0
RATIO_BAND_GT_OCCLUDED = (1.20, 8.00)
RATIO_BAND_GT_VISIBLE = (1.20, 3.00)

# ── Badge visibility from the SEGMENTATION buffer (graded, visual-mesh based) ──
# A ray test is unusable here: it hits the VHACD collision hull, not the visual
# surface, which reported clearly-visible badges on shoes/bowls as hidden.
BADGE_HIDDEN_MAX = 0.15    # "must be revealed": at most this fraction visible
BADGE_READABLE_MIN = 0.85  # "readable now": at least this fraction visible

# The benchmark's own scene gate for A/B question types
# (core/worker.py::_MIN_OCCLUSION_AB): at least one object must be this
# occluded, otherwise the scene is too easy to require manipulation.
MIN_OCCLUSION_AB = 0.60

# A badge the seg buffer says is visible must occupy at least this fraction of
# achromatic bright/dark pixels in the RENDERED image, else the episode is
# rejected (the bake and the renderer disagreed about which face is up).
BADGE_RENDERED_MIN = 0.25

# Assets whose badge does not render correctly in the full pipeline even though
# it works in isolation.  Verified by human spot-check of the badge montage:
# 13/14 unoccluded labels rendered correctly; this one was the sole failure
# (the white-and-red variant of the same product is fine).  Cheaper to exclude
# than to chase renderer-state-specific behaviour.
BADGE_ASSET_DENYLIST = {
    "dell_ink_cartridge_yellow_31",
}

# Minimum fraction of the badge's EXPECTED on-screen disc area that must
# actually change when the texture is applied.  Catches badges painted onto
# steeply foreshortened or otherwise unusable surfaces, which report a healthy
# badge_cm_actual while rendering as a sliver.
BADGE_SCREEN_MIN = 0.35

# After the badge sprite has been composited and clipped to its UV island,
# this much of the sprite's own non-transparent area must remain.  This is a
# post-bake invariant: it catches a programming/UV failure before a badged
# scene can be accepted.  It is shape agnostic (a triangle is compared with
# its own alpha mask, not with a circumscribed circle).
BADGE_PAINT_MIN = 0.60
