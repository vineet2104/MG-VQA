"""
Scene Manager — Blender scene operations.

Handles mesh import, transforms, camera, lighting, rendering.
Runs inside Blender's Python. All functions called from main thread.
"""

import bpy
import os
import math
import mathutils

# Registry: (body_id, link_index) → blender object name
_registry: dict[tuple[int, int], str] = {}


def _key(body_id: int, link_index: int) -> tuple[int, int]:
    return (body_id, link_index)


def _quat_pb2bl(xyzw: list[float]) -> mathutils.Quaternion:
    """PyBullet (x,y,z,w) → Blender (w,x,y,z)."""
    x, y, z, w = xyzw
    return mathutils.Quaternion((w, x, y, z))


def clear_scene():
    """Nuke everything, start fresh."""
    global _registry
    _registry.clear()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")


def remove_bodies(params: dict) -> dict:
    """Remove specific bodies from the Blender scene by body_id.

    Incremental removal — avoids the expensive full clear+reload cycle
    that ``_sync_scene`` previously used when bodies were removed from
    PyBullet.  Each body may have multiple links; all associated Blender
    objects and their meshes/materials are deleted.

    Expects:
        body_ids: list[int] — PyBullet body IDs to remove.
    """
    global _registry
    body_ids = set(params.get("body_ids", []))
    removed = 0

    # Collect registry keys and Blender object names for these bodies
    keys_to_remove = []
    for key, obj_name in list(_registry.items()):
        bid, _ = key
        if bid in body_ids:
            keys_to_remove.append((key, obj_name))

    for key, obj_name in keys_to_remove:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            # Remove mesh data too (otherwise it lingers as orphan)
            mesh = obj.data if obj.type == 'MESH' else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            removed += 1
        del _registry[key]

    return {"removed": removed, "body_ids": list(body_ids)}


def _make_material(name: str, rgba: list[float], category: str = "object") -> bpy.types.Material:
    """Create a simple Principled BSDF material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = tuple(rgba)

    if category == "robot":
        bsdf.inputs["Metallic"].default_value = 0.85
        bsdf.inputs["Roughness"].default_value = 0.35
        # Default robot color: dark gunmetal gray (if PyBullet sends white)
        if abs(rgba[0] - 1.0) < 0.01 and abs(rgba[1] - 1.0) < 0.01 and abs(rgba[2] - 1.0) < 0.01:
            bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.18, 1.0)
    elif category == "floor":
        bsdf.inputs["Roughness"].default_value = 0.8
    elif category == "table":
        bsdf.inputs["Roughness"].default_value = 0.6
    else:
        # Generic objects: matte diffuse, close to PyBullet's Lambertian look
        bsdf.inputs["Roughness"].default_value = 0.9
        bsdf.inputs["Specular IOR Level"].default_value = 0.1

    return mat


def _make_vertex_color_material(name: str, layer_name: str) -> bpy.types.Material:
    """Principled BSDF driven by a mesh COLOR ATTRIBUTE (per-vertex RGB).

    Many BOP meshes (hb_, itodd_, xyzibd_, lmo_, …) ship colour ONLY as
    baked per-vertex colours with no .mtl / texture image.  Blender imports
    those as a colour attribute (e.g. 'Color') but, with no material, the
    scene builder used to fall back to PyBullet's flat white RGBA — so the
    object rendered as a grey blob (e.g. a "dark green and black" drill).
    Wiring the colour attribute into Base Color renders the true colours.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.location = (-260, 0)
    vcol.layer_name = layer_name
    links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Matte, PyBullet-like look (consistent with generic objects).
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular IOR Level"].default_value = 0.1
    return mat


def _mesh_color_attr_name(obj: bpy.types.Object) -> str | None:
    """Return the mesh's colour-attribute name (per-vertex RGB), or None."""
    me = getattr(obj, "data", None)
    if me is None:
        return None
    cas = getattr(me, "color_attributes", None)
    if cas:
        return cas[0].name
    # legacy fallback (Blender < 3.2)
    vcs = getattr(me, "vertex_colors", None)
    if vcs:
        return vcs[0].name
    return None


def _make_checker_material(
    name: str,
    color1=(0.9, 0.9, 0.95, 1),
    color2=(0.55, 0.65, 0.85, 1),
    plane_size: float = 200.0,
):
    """Checkerboard material with subtle blurred HDRI reflections (no Fresnel).

    Uses UV coordinates (0-1 across the plane) with scale computed from
    the actual plane size to produce 1-unit checker cells matching PyBullet.
    
    The material uses:
    - Diffuse BSDF for the checker pattern (matte appearance)
    - Glossy BSDF with very high roughness sampling the HDRI (very blurred reflections)
    - CONSTANT mix factor (no Fresnel) for uniform subtle reflections
    
    ShaderNodeTexChecker Scale=S → each cell is 1/(2*S) UV units wide.
    For 1-unit cells on a plane_size-unit plane: S = plane_size / 2.
    """
    checker_scale = plane_size / 2.0

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (800, 0)

    # --- Diffuse component (checker pattern) ---
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (300, 200)
    diffuse.inputs["Roughness"].default_value = 0.0

    # Checker texture
    checker = nodes.new("ShaderNodeTexChecker")
    checker.location = (-100, 200)
    checker.inputs["Scale"].default_value = checker_scale
    checker.inputs["Color1"].default_value = color1
    checker.inputs["Color2"].default_value = color2
    links.new(checker.outputs["Color"], diffuse.inputs["Color"])

    # UV coordinates for checker
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-400, 200)
    links.new(texcoord.outputs["UV"], checker.inputs["Vector"])

    # --- Glossy component (very blurred HDRI reflections) ---
    glossy = nodes.new("ShaderNodeBsdfGlossy")
    glossy.location = (300, -100)
    # Very high roughness = extremely blurred reflections
    glossy.inputs["Roughness"].default_value = 0.95

    # Sample the world HDRI for glossy color
    # We use the reflection vector to sample the environment
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (-100, -100)
    
    # Try to use the same HDRI as the world background
    world = bpy.context.scene.world
    hdri_image = None
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == "TEX_ENVIRONMENT" and node.image:
                hdri_image = node.image
                break
    
    if hdri_image:
        env_tex.image = hdri_image
    
    # Use reflection vector to sample environment
    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-400, -100)
    links.new(geometry.outputs["Reflection"], env_tex.inputs["Vector"])
    
    # Connect HDRI to glossy color
    links.new(env_tex.outputs["Color"], glossy.inputs["Color"])

    # --- Mix diffuse and glossy with CONSTANT factor (NO Fresnel) ---
    mix_shader = nodes.new("ShaderNodeMixShader")
    mix_shader.location = (550, 0)
    
    # Use a constant mix factor - NO Fresnel, uniform reflection everywhere
    # Very low value (0.03-0.08) for subtle uniform reflections
    mix_shader.inputs["Fac"].default_value = 0.05
    
    # Connect: constant mix, mostly diffuse with tiny bit of glossy
    links.new(diffuse.outputs["BSDF"], mix_shader.inputs[1])  # First shader (diffuse)
    links.new(glossy.outputs["BSDF"], mix_shader.inputs[2])   # Second shader (glossy)
    
    # Connect to output
    links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])

    return mat


def _classify(body_name: str, link_name: str) -> str:
    """Classify a (body, link) pair as robot / table / floor / object.

    Uses tight matching so spawned assets whose names happen to contain
    a robot/table keyword (e.g. ``"panda_android_figure"``,
    ``"wooden_toy_robot"``, ``"purple_dollhouse_table"``) are NOT
    misclassified.

    Strategy:
      * **Unambiguous robot tokens** (``ur5``, ``robotiq``, ``kuka``,
        ``lbr``, ``iiwa``, ``banchetto``) — if *any* of these appears
        as a token, the body is a robot.
      * **Ambiguous tokens** (``panda``, ``robot``, ``gripper``, etc.)
        — only match if they appear **without** object-indicator tokens
        (``figure``, ``figurine``, ``toy``, ``pencil``, ``case``,
        ``bead``, ``dollhouse``, ``caddy``, ``organizer``, …).
      * **Table / floor**: only match when the body name is *exactly*
        ``"table"`` (1 token) or starts with an unambiguous prefix
        (``"plane.urdf"``).

    The real safety net is ``BlenderRenderer.infrastructure_body_ids``
    (populated by the env) which is consulted *before* ``_classify``
    for the hide/show decision.  ``_classify`` only affects **material
    assignment** — robot → gunmetal, table → wood, etc.
    """
    import re as _re
    # Tokenize body and link separately — link names like ``"link"`` or
    # ``"base_link"`` are generic PyBullet defaults and must not pollute
    # the robot-keyword check.
    body_lower = body_name.lower().strip()
    link_lower = link_name.lower().strip()
    body_tokens = set(_re.split(r"[_/\-\s]+", body_lower))
    body_tokens.discard("")
    all_tokens = body_tokens | set(_re.split(r"[_/\-\s]+", link_lower))
    all_tokens.discard("")

    # ── Unambiguous robot identifiers (never appear in asset names) ──
    unambiguous_robot = {"ur5", "ur10", "robotiq", "kuka", "lbr",
                         "iiwa", "banchetto"}
    if all_tokens & unambiguous_robot:
        return "robot"

    # ── Object-indicator tokens: if ANY of these appear in the
    #    **body** name alongside an ambiguous keyword, the body is a
    #    spawned object, NOT infrastructure.
    object_indicators = {
        "figure", "figurine", "toy", "pencil", "case", "bead",
        "dollhouse", "doll", "caddy", "organizer", "android",
        "wooden", "blue", "purple", "bottle", "box", "cream",
        "supplement", "powder", "pasta", "sauce",
    }
    has_object_hint = bool(body_tokens & object_indicators)

    # ── Ambiguous robot tokens (checked in body_tokens only) ─────────
    ambiguous_robot = {"robot", "panda", "gripper", "finger",
                       "link", "arm", "joint"}
    if (body_tokens & ambiguous_robot) and not has_object_hint:
        return "robot"

    # ── Table / desk ─────────────────────────────────────────────────
    # Classify as "table" when the body name is short (≤2 tokens) and
    # contains "table" or "desk" with no object hints.  Multi-word
    # asset names like ``"purple_dollhouse_table"`` are caught by the
    # object-indicator guard.
    if ("table" in body_tokens or "desk" in body_tokens) and not has_object_hint:
        if len(body_tokens) <= 3:
            return "table"

    # ── Floor / ground ───────────────────────────────────────────────
    if body_tokens & {"floor", "plane", "ground"}:
        if not has_object_hint:
            return "floor"

    return "object"


def _import_mesh(mesh_path: str, mesh_type: str) -> bpy.types.Object | None:
    """Import a mesh file, return the resulting mesh object."""
    if not mesh_path or not os.path.isfile(mesh_path):
        return None

    existing = set(bpy.data.objects.keys())

    if mesh_type == "obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif mesh_type == "stl":
        bpy.ops.wm.stl_import(filepath=mesh_path)
    elif mesh_type == "dae":
        # Blender 5.x removed collada_import.  The operator proxy still
        # passes hasattr() so we must try/except the actual call.
        try:
            bpy.ops.wm.collada_import(filepath=mesh_path)
        except (AttributeError, RuntimeError):
            print(f"[scene_manager] WARNING: Collada (.dae) import not available "
                  f"in Blender {bpy.app.version_string}. Skipping: {mesh_path}")
            return None
    else:
        return None

    new_objs = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in existing]
    mesh_objs = [o for o in new_objs if o.type == "MESH"]

    # Remove non-mesh (cameras/lights from DAE)
    for o in new_objs:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)

    if not mesh_objs:
        return None

    # Join multiple mesh objects into one
    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()
    
    result = bpy.context.view_layer.objects.active if len(mesh_objs) > 1 else mesh_objs[0]

    # Fix inverted normals — many OBJ/STL meshes from PyBullet assets
    # have inward-facing normals.  We detect this by checking whether the
    # majority of face normals point away from the mesh centroid.
    _fix_normals(result)

    return result


def _fix_normals(obj):
    """Fix normals on imported meshes so they face outward.

    Many OBJ/STL files carry custom split normals from their source
    application.  These can be inverted, and Blender/EEVEE will use them
    for shading even if the face normals are correct.

    Strategy:
    1. Clear any custom split normals (so Blender uses face normals).
    2. Recalculate face normals outward via ``normals_make_consistent``.
    3. Flip remaining inward faces individually via bmesh (for non-manifold).
    4. Disable backface culling on all materials as safety net.
    """
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    if not mesh.polygons:
        return

    # Step 0: clear custom split normals — they override face normals
    # and are often inverted in PyBullet asset OBJ/STL files.
    if mesh.has_custom_normals:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        bpy.ops.object.mode_set(mode="OBJECT")

    # Step 1: recalculate normals outside (works well for manifold meshes)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Step 2: flip remaining inward-facing faces via bmesh
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    if bm.verts:
        cx = sum(v.co.x for v in bm.verts) / len(bm.verts)
        cy = sum(v.co.y for v in bm.verts) / len(bm.verts)
        cz = sum(v.co.z for v in bm.verts) / len(bm.verts)
        centroid = mathutils.Vector((cx, cy, cz))

        flipped = 0
        for face in bm.faces:
            outward = face.calc_center_median() - centroid
            if outward.dot(face.normal) < 0:
                face.normal_flip()
                flipped += 1

        if flipped > 0:
            bm.to_mesh(mesh)
            mesh.update()

    bm.free()

    # Step 3: disable backface culling on all materials as safety net
    for mat in obj.data.materials:
        if mat:
            mat.use_backface_culling = False


def load_body(params: dict) -> dict:
    """Import or create a body in the Blender scene."""
    body_id = params["body_id"]
    link_index = params["link_index"]
    mesh_path = params.get("mesh_path", "")
    mesh_type = params.get("mesh_type", "obj")
    position = params.get("position", [0, 0, 0])
    orn_xyzw = params.get("orientation_xyzw", [0, 0, 0, 1])
    scale = params.get("scale", [1, 1, 1])
    rgba = params.get("rgba", [0.5, 0.5, 0.5, 1.0])
    body_name = params.get("body_name", "")
    link_name = params.get("link_name", "")
    prim_type = params.get("primitive_type", "")
    prim_dims = params.get("primitive_dims", [])

    key = _key(body_id, link_index)
    category = _classify(body_name, link_name)
    obj_name = body_name or f"body{body_id}"
    if link_name:
        obj_name += f"_{link_name}"

    obj = None

    if mesh_type == "primitive" or (not mesh_path and prim_type):
        if prim_type == "box":
            sx = prim_dims[0] if len(prim_dims) > 0 else 0.1
            sy = prim_dims[1] if len(prim_dims) > 1 else sx
            sz = prim_dims[2] if len(prim_dims) > 2 else sx
            # cube_add(size=1) creates a cube from -0.5 to +0.5 on each axis.
            # Scale sets the final dimensions; no transform_apply so that
            # location remains independent for later update_transform calls.
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=position)
            obj = bpy.context.object
            obj.scale = (sx, sy, sz)
        elif prim_type == "sphere":
            r = prim_dims[0] if prim_dims else 0.05
            bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=position, segments=32, ring_count=16)
            obj = bpy.context.object
        elif prim_type == "cylinder":
            r = prim_dims[0] if prim_dims else 0.05
            h = prim_dims[1] if len(prim_dims) > 1 else 0.1
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=position, vertices=32)
            obj = bpy.context.object
        elif prim_type in ("checker_plane", "plain_plane"):
            sx = prim_dims[0] if len(prim_dims) > 0 else 200
            sy = prim_dims[1] if len(prim_dims) > 1 else sx
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=position)
            obj = bpy.context.object
            obj.scale = (sx, sy, 1)
            bpy.ops.object.transform_apply(scale=True)
            if prim_type == "checker_plane":
                # Use checker material — pass plane size for correct UV-based cell sizing
                cmat = _make_checker_material(f"mat_{obj_name}", plane_size=max(sx, sy))
            else:
                # Plain neutral surface — no checkerboard pattern
                cmat = bpy.data.materials.new(name=f"mat_{obj_name}_plain")
                cmat.use_nodes = True
                bsdf = cmat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (0.25, 0.25, 0.25, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.9
            obj.data.materials.clear()
            obj.data.materials.append(cmat)
            obj.name = obj_name
            if obj.data:
                obj.data.name = f"{obj_name}_mesh"
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = _quat_pb2bl(orn_xyzw)
            _registry[key] = obj.name
            return {"object_name": obj.name}
        else:
            bpy.ops.mesh.primitive_cube_add(size=0.05, location=position)
            obj = bpy.context.object
    else:
        obj = _import_mesh(mesh_path, mesh_type)

    if obj is None:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=position)
        obj = bpy.context.object

    # Name
    obj.name = obj_name
    if obj.data:
        obj.data.name = f"{obj_name}_mesh"

    # Transform: position was already set for primitives at creation,
    # but for imported meshes we need to set it
    if mesh_path:
        obj.location = mathutils.Vector(position)
        obj.scale = mathutils.Vector(scale)

    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = _quat_pb2bl(orn_xyzw)

    # Material — apply PyBullet's RGBA color unless the object already has
    # textured materials from the imported mesh.  PyBullet reports
    # rgba=(1,1,1,1) for texture-mapped meshes, so treat that as "keep
    # existing materials".
    has_existing_textures = False
    if obj.data.materials:
        for mat in obj.data.materials:
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image:
                        # Check that the texture file actually exists/loaded
                        img = node.image
                        if img.packed_file or (img.filepath and os.path.isfile(bpy.path.abspath(img.filepath))):
                            has_existing_textures = True
                            break
            if has_existing_textures:
                break

    is_white_rgba = (abs(rgba[0] - 1.0) < 0.01 and
                     abs(rgba[1] - 1.0) < 0.01 and
                     abs(rgba[2] - 1.0) < 0.01)

    # Per-vertex colour meshes (BOP: hb_/itodd_/xyzibd_/lmo_ …) carry colour
    # ONLY as a mesh colour attribute (no .mtl, no texture) — detect it so we
    # can drive Base Color from it instead of PyBullet's flat white RGBA.
    vcol_layer = None
    if (not prim_type and category not in ("robot", "floor", "table")
            and not has_existing_textures):
        vcol_layer = _mesh_color_attr_name(obj)

    if has_existing_textures and is_white_rgba and category != "robot":
        # Keep the imported textured materials as-is
        for mat in obj.data.materials:
            if mat:
                mat.use_backface_culling = False
    elif vcol_layer and (not obj.data.materials or is_white_rgba):
        # Per-vertex colour mesh with no usable material → render its baked
        # vertex colours (fixes BOP objects rendering as grey/white blobs).
        mat = _make_vertex_color_material(f"mat_{obj_name}_vcol", vcol_layer)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    elif prim_type or not obj.data.materials or category == "robot":
        # Primitives or meshes with no materials: create from scratch
        mat = _make_material(f"mat_{obj_name}", rgba, category)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    else:
        # Imported mesh with existing materials: set the base color
        # to match PyBullet's RGBA while keeping other PBR properties
        for mat in obj.data.materials:
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        node.inputs["Base Color"].default_value = tuple(rgba)
                        if category == "robot":
                            node.inputs["Metallic"].default_value = 0.85
                            node.inputs["Roughness"].default_value = 0.35
                        break

    # Smooth shading — only for curved surfaces, not boxes/planes
    is_flat_prim = prim_type in ("box", "checker_plane", "")
    if not is_flat_prim:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            pass

    _registry[key] = obj.name
    return {"object_name": obj.name}


def update_transform(params: dict) -> dict:
    """Move an existing object."""
    key = _key(params["body_id"], params["link_index"])
    obj_name = _registry.get(key)
    if not obj_name or obj_name not in bpy.data.objects:
        return {"error": f"Not found: {key}"}

    obj = bpy.data.objects[obj_name]
    obj.location = mathutils.Vector(params["position"])
    obj.rotation_quaternion = _quat_pb2bl(params["orientation_xyzw"])
    return {"object_name": obj_name}


def batch_update_transforms(params: dict) -> dict:
    """Update multiple transforms."""
    results = [update_transform(u) for u in params.get("updates", [])]
    return {"count": len(results)}


def set_camera(params: dict) -> dict:
    """
    Set camera to match PyBullet's computeViewMatrix output exactly.

    PyBullet/OpenGL camera: looks down -Z, Y is up.
    Blender camera: same convention — looks down local -Z, Y is up.
    So we just build the look-at rotation matrix and apply it.
    """
    eye = mathutils.Vector(params["eye"])
    target = mathutils.Vector(params["target"])
    up = mathutils.Vector(params.get("up", [0, 0, 1]))
    fov = params.get("fov", 60.0)
    near = params.get("near", 0.1)
    far = params.get("far", 100.0)
    res = params.get("resolution", [1280, 720])

    # Find or create camera
    cam_obj = None
    for o in bpy.data.objects:
        if o.type == "CAMERA":
            cam_obj = o
            break
    if cam_obj is None:
        bpy.ops.object.camera_add()
        cam_obj = bpy.context.object

    cam_obj.name = "BridgeCamera"
    bpy.context.scene.camera = cam_obj

    # Build look-at: same math as OpenGL gluLookAt
    forward = (target - eye).normalized()
    right = forward.cross(up).normalized()
    if right.length < 1e-6:
        right = mathutils.Vector((1, 0, 0))
    cam_up = right.cross(forward).normalized()

    # Blender camera local axes: +X=right, +Y=up, -Z=forward
    # Each row is a local axis direction expressed in world coords
    rot = mathutils.Matrix((
        (right.x,    cam_up.x,   -forward.x),
        (right.y,    cam_up.y,   -forward.y),
        (right.z,    cam_up.z,   -forward.z),
    ))

    cam_obj.location = eye
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.rotation_quaternion = rot.to_quaternion()

    # Lens — PyBullet FOV is vertical; set sensor_fit so Blender interprets
    # camera.data.angle as vertical FOV too.
    cam_obj.data.sensor_fit = "VERTICAL"
    cam_obj.data.lens_unit = "FOV"
    cam_obj.data.angle = math.radians(fov)
    cam_obj.data.clip_start = near
    cam_obj.data.clip_end = max(far, 10.0)

    # Resolution
    bpy.context.scene.render.resolution_x = res[0]
    bpy.context.scene.render.resolution_y = res[1]
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = 1.0

    return {"camera": cam_obj.name}


def _setup_gpu():
    """Enable GPU rendering if available."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for ct in ["OPTIX", "CUDA", "HIP"]:
            try:
                prefs.compute_device_type = ct
                prefs.get_devices()
                gpu = [d for d in prefs.devices if d.type != "CPU"]
                if gpu:
                    for d in prefs.devices:
                        d.use = True
                    print(f"[GPU] Using {ct}: {[d.name for d in gpu]}")
                    return
            except Exception:
                continue
    except Exception:
        pass
    print("[GPU] No GPU found, using CPU")


def configure_lighting(params: dict) -> dict:
    """Set up lighting.

    Presets:
        studio    — Blender's built-in studio.exr HDRI
        interior  — interior.exr HDRI
        courtyard — courtyard.exr HDRI
        city      — city.exr HDRI
        forest    — forest.exr HDRI
        sunset    — sunset.exr HDRI
        sunrise   — sunrise.exr HDRI
        night     — night.exr HDRI
        pybullet_match — flat ambient (no HDRI)

    HDRI presets use the environment map for both illumination and
    the visible background.

    Also accepts hdri_path for a custom .exr/.hdr file.

    Optional:
        add_directional_light (bool): If True, add a Sun lamp alongside
            the HDRI to cast sharp shadows.  This is the "hard" lighting
            mode — objects will have visible cast shadows on the table,
            making the scene more visually challenging for the VLM.
    """
    preset = params.get("preset", "studio")
    mult = params.get("intensity_multiplier", 1.0)
    hdri_path = params.get("hdri_path", "")
    add_directional = params.get("add_directional_light", False)

    # Remove existing lights
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    if preset == "pybullet_match":
        _setup_pybullet_match_lights(mult)
    else:
        _setup_hdri_lighting(preset, mult, hdri_path)

    if add_directional:
        _add_directional_sun(mult)

    return {"preset": preset, "add_directional_light": add_directional}


def _add_directional_sun(mult: float):
    """Add a Sun lamp for directional lighting with cast shadows.

    Placed at 40° elevation from front-left to cast visible shadows
    without washing out the HDRI ambient.  Strength is scaled by the
    global intensity multiplier so easy/hard modes stay proportional.

    The Sun direction in Blender is determined by the lamp object's
    rotation, not its position (Sun lamps have infinite distance).
    We point it roughly toward the table centre from front-left.
    """
    import mathutils
    sun_data = bpy.data.lights.new(name="DirectionalSun", type="SUN")
    sun_data.energy = 3.0 * mult        # moderate strength alongside HDRI
    sun_data.angle = math.radians(2.0)   # 2° angular diameter → soft penumbra
    sun_data.use_shadow = True

    sun_obj = bpy.data.objects.new("DirectionalSun", sun_data)
    bpy.context.collection.objects.link(sun_obj)

    # Rotation: point the sun toward the table from front-left, ~40° elev.
    # Euler XYZ: X = elevation tilt from horizontal, Z = azimuth rotation
    sun_obj.rotation_euler = mathutils.Euler(
        (math.radians(40), math.radians(10), math.radians(-30)), 'XYZ')
    # Position doesn't matter for Sun lamps (rays are parallel), but set
    # it above the scene so it's visible when inspecting the .blend file.
    sun_obj.location = (0.0, -1.0, 3.0)
    print(f"[Lighting] Added directional Sun lamp (energy={sun_data.energy:.1f}, "
          f"angle={math.degrees(sun_data.angle):.1f}°)")


# Map preset names to Blender's bundled world HDRI files
_HDRI_PRESETS = {
    "studio", "interior", "courtyard", "city",
    "forest", "sunset", "sunrise", "night",
}


def _find_blender_hdri(name: str) -> str:
    """Find a bundled Blender world HDRI by name."""
    # Blender stores them relative to the executable
    blender_exe = bpy.app.binary_path
    blender_dir = os.path.dirname(blender_exe)
    ver = f"{bpy.app.version[0]}.{bpy.app.version[1]}"
    hdri_dir = os.path.join(blender_dir, ver, "datafiles", "studiolights", "world")
    path = os.path.join(hdri_dir, f"{name}.exr")
    if os.path.isfile(path):
        return path
    # Fallback: search
    for f in os.listdir(hdri_dir) if os.path.isdir(hdri_dir) else []:
        if f.lower().startswith(name.lower()) and f.endswith(".exr"):
            return os.path.join(hdri_dir, f)
    return ""


def _setup_hdri_lighting(preset: str, mult: float, hdri_path: str = ""):
    """HDRI environment lighting.

    The HDRI provides both scene illumination and the visible background.
    """
    # Resolve HDRI path
    if hdri_path and os.path.isfile(hdri_path):
        env_path = hdri_path
    elif preset in _HDRI_PRESETS:
        env_path = _find_blender_hdri(preset)
    else:
        env_path = _find_blender_hdri("studio")

    # Set up world environment
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputWorld")
    out.location = (600, 0)
    bg = nodes.new("ShaderNodeBackground")
    bg.location = (300, 0)
    bg.inputs["Strength"].default_value = 1.0 * mult
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    if env_path and os.path.isfile(env_path):
        env_tex = nodes.new("ShaderNodeTexEnvironment")
        env_tex.location = (0, 0)
        env_tex.image = bpy.data.images.load(env_path, check_existing=True)
        links.new(env_tex.outputs["Color"], bg.inputs["Color"])
        print(f"[Lighting] HDRI: {env_path}")
    else:
        bg.inputs["Color"].default_value = (0.8, 0.82, 0.85, 1.0)
        print("[Lighting] No HDRI found, using flat ambient")


def set_world_hdri(params: dict) -> dict:
    """Force the world background to a specific HDRI/image file.

    Used by paper-figure scripts that want a guaranteed visible HDRI
    background (configure_lighting's preset machinery sometimes results
    in an invisible/dim background).  This handler:

      1. Rebuilds the world shader graph from scratch.
      2. Loads the image with explicit colorspace (Linear) for HDRI.
      3. Connects: TexEnvironment → Background → World Output.
      4. Sets background ``Strength`` to ``strength`` directly (no mult).
      5. Optionally rotates the HDRI around Z by ``rotation_deg``.

    Args:
        image_path: Absolute path to .hdr / .exr / .jpg / .png.
        strength: Background emission strength (1.0 = unmodified).
        rotation_deg: Rotation around Z axis in degrees.

    Returns:
        ``{"applied": <path>, "strength": <float>}`` on success.
    """
    image_path = params.get("image_path", "")
    strength = float(params.get("strength", 1.0))
    rotation_deg = float(params.get("rotation_deg", 0.0))

    if not image_path or not os.path.isfile(image_path):
        return {"error": f"image not found: {image_path}"}

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    out  = nodes.new("ShaderNodeOutputWorld");  out.location  = (700, 0)
    bg   = nodes.new("ShaderNodeBackground");   bg.location   = (450, 0)
    tex  = nodes.new("ShaderNodeTexEnvironment"); tex.location = (200, 0)
    mapp = nodes.new("ShaderNodeMapping");      mapp.location = (-50, 0)
    coord = nodes.new("ShaderNodeTexCoord");    coord.location = (-300, 0)

    # Load the image. For HDR/EXR Blender treats them as linear
    # automatically; for LDR JPG/PNG we tell it to treat as sRGB so colours
    # don't double-gamma.
    img = bpy.data.images.load(image_path, check_existing=True)
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".hdr", ".exr"):
        try: img.colorspace_settings.name = "Linear Rec.709"
        except Exception:
            try: img.colorspace_settings.name = "Linear"
            except Exception: pass
    else:
        try: img.colorspace_settings.name = "sRGB"
        except Exception: pass
    tex.image = img

    # Optional Z rotation
    import math as _math
    mapp.inputs["Rotation"].default_value[2] = _math.radians(rotation_deg)

    # Wire: Generated coords → Mapping → Env Texture → Background → Output
    links.new(coord.outputs["Generated"], mapp.inputs["Vector"])
    links.new(mapp.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    bg.inputs["Strength"].default_value = strength

    # Make sure the world is actually visible to the camera in BOTH engines.
    # Blender 5.x sometimes leaves film transparent on, which hides the
    # world background entirely.
    try:
        bpy.context.scene.render.film_transparent = False
    except Exception:
        pass
    # Cycles visibility flags — harmless on EEVEE.
    try:
        world.cycles_visibility.camera = True
        world.cycles_visibility.diffuse = True
        world.cycles_visibility.glossy = True
    except Exception:
        pass
    # Force the world data block to update (some EEVEE versions cache the
    # compiled shader and need an explicit dirty flag to re-link).
    try:
        world.update_tag()
        bpy.context.view_layer.update()
    except Exception:
        pass

    # Diagnostic info
    img_size = (img.size[0], img.size[1]) if img.has_data else (0, 0)
    print(f"[set_world_hdri] applied {image_path} "
          f"(size={img_size}, strength={strength}, rot_deg={rotation_deg})")
    return {
        "applied": image_path,
        "strength": strength,
        "image_size": list(img_size),
        "has_data": bool(img.has_data),
    }


def _setup_pybullet_match_lights(mult: float):
    """Flat ambient lighting: no HDRI, no shadows, neutral grey background.

    Strength 0.8 × mult keeps object colours readable while remaining
    shadow-free (no directional component).
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    out.location = (300, 0)
    bg = nodes.new("ShaderNodeBackground")
    bg.location = (0, 0)
    bg.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)
    bg.inputs["Strength"].default_value = 0.8 * mult
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def save_blend(params: dict) -> dict:
    """Save the current scene to a .blend file for inspection.

    All external resources (textures, HDRIs, etc.) are packed into the
    .blend so it can be opened on a different machine without missing files.
    """
    out_path = os.path.abspath(params["output_path"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Pack all external images (HDRIs, mesh textures, etc.) into the .blend
    # so the file is fully self-contained and portable.
    for img in bpy.data.images:
        if img.source in ("FILE", "SEQUENCE") and not img.packed_file:
            try:
                img.pack()
            except Exception as e:
                print(f"[save_blend] Could not pack image '{img.name}': {e}")

    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    return {"output_path": out_path}


def set_table_texture(params: dict) -> dict:
    """Apply a PBR texture set to table/tabletop objects in the scene.

    Expects a directory containing CCO-style PBR texture maps:
      - *_Color.jpg       → Base Color
      - *_Roughness.jpg   → Roughness
      - *_Normal.jpg      → Normal Map
      - *_AmbientOcclusion.jpg → (mixed into Base Color)

    Uses Generated coordinates with BOX projection so that primitive box
    meshes (which lack proper UVs) display textures correctly without
    needing UV unwrapping or transform_apply (which would interfere with
    the renderer's transform management).

    params:
        texture_dir: Absolute path to the texture directory.
        uv_scale: UV tiling scale (default 2.0 — tiles the texture 2x).
    """
    texture_dir = params["texture_dir"]
    uv_scale = params.get("uv_scale", 2.0)

    if not os.path.isdir(texture_dir):
        return {"error": f"Texture dir not found: {texture_dir}"}

    # Find texture maps by suffix
    import glob as _glob
    def _find(pattern):
        hits = _glob.glob(os.path.join(texture_dir, pattern))
        return hits[0] if hits else None

    color_path = _find("*_Color.*") or _find("*_Diffuse.*")
    rough_path = _find("*_Roughness.*")
    normal_path = _find("*_Normal.*") or _find("*_norm.*")
    ao_path = _find("*_AmbientOcclusion.*")

    if not color_path:
        return {"error": f"No Color texture found in {texture_dir}"}

    print(f"[set_table_texture] Applying texture from {os.path.basename(texture_dir)}")

    # Build a PBR material
    mat = bpy.data.materials.new(name="TablePBR")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Core nodes
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Use Generated coordinates — they map the bounding box to 0–1 range
    # and work on primitives without any UV unwrapping needed.
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (uv_scale, uv_scale, uv_scale)
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

    # Color texture — BOX projection maps all 6 faces correctly
    # without needing UV unwrapping (which would require transform_apply
    # and break the renderer's transform management).
    color_tex = nodes.new("ShaderNodeTexImage")
    color_tex.image = bpy.data.images.load(color_path)
    color_tex.projection = "BOX"
    color_tex.projection_blend = 0.2
    links.new(mapping.outputs["Vector"], color_tex.inputs["Vector"])

    if ao_path:
        # Multiply color × AO
        ao_tex = nodes.new("ShaderNodeTexImage")
        ao_tex.image = bpy.data.images.load(ao_path)
        ao_tex.image.colorspace_settings.name = "Non-Color"
        ao_tex.projection = "BOX"
        ao_tex.projection_blend = 0.2
        links.new(mapping.outputs["Vector"], ao_tex.inputs["Vector"])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1.0  # factor
        links.new(color_tex.outputs["Color"], mix.inputs[6])
        links.new(ao_tex.outputs["Color"], mix.inputs[7])
        links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    else:
        links.new(color_tex.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness texture
    if rough_path:
        rough_tex = nodes.new("ShaderNodeTexImage")
        rough_tex.image = bpy.data.images.load(rough_path)
        rough_tex.image.colorspace_settings.name = "Non-Color"
        rough_tex.projection = "BOX"
        rough_tex.projection_blend = 0.2
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = 0.5

    # Normal map
    if normal_path:
        norm_tex = nodes.new("ShaderNodeTexImage")
        norm_tex.image = bpy.data.images.load(normal_path)
        norm_tex.image.colorspace_settings.name = "Non-Color"
        norm_tex.projection = "BOX"
        norm_tex.projection_blend = 0.2
        links.new(mapping.outputs["Vector"], norm_tex.inputs["Vector"])
        norm_map = nodes.new("ShaderNodeNormalMap")
        norm_map.inputs["Strength"].default_value = 1.0
        links.new(norm_tex.outputs["Color"], norm_map.inputs["Color"])
        links.new(norm_map.outputs["Normal"], bsdf.inputs["Normal"])

    # Apply to all table/tabletop objects
    # Apply to all table surfaces and the robot stand (banchetto)
    # so all large flat surfaces share the same wood texture.
    # Exclude the tray (URDF name "tabletop") — it keeps its default grey.
    applied_to = []
    for obj_name in list(bpy.data.objects.keys()):
        name_lower = obj_name.lower()
        # Skip the tray: its Blender name is "tabletop_base" (from URDF "tabletop")
        if "tabletop" in name_lower:
            continue
        if ("table" in name_lower
                or "banchetto" in name_lower or "ur5_stand" in name_lower):
            obj = bpy.data.objects[obj_name]
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                applied_to.append(obj_name)

    print(f"  Applied to: {applied_to}")
    return {"applied_to": applied_to, "texture_dir": texture_dir}


def render_frame(params: dict) -> dict:
    """Render current scene to file."""
    out_path = os.path.abspath(params["output_path"])
    samples = params.get("samples", 64)
    engine = params.get("engine", "CYCLES")
    denoise = params.get("denoise", True)
    hero_quality = params.get("hero_quality", False)
    film_transparent = params.get("film_transparent", False)
    color_depth = params.get("color_depth", "8")
    resolution = params.get("resolution")  # optional [W, H]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _configure_render(engine, samples, denoise,
                       hero_quality=hero_quality,
                       film_transparent=film_transparent,
                       color_depth=color_depth)
    if resolution is not None:
        bpy.context.scene.render.resolution_x = int(resolution[0])
        bpy.context.scene.render.resolution_y = int(resolution[1])
        bpy.context.scene.render.resolution_percentage = 100

    bpy.context.scene.render.filepath = out_path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"

    bpy.ops.render.render(write_still=True)
    return {"output_path": out_path}


def render_frame_bytes(params: dict) -> dict:
    """Render and return raw RGBA pixel data as base64."""
    import base64
    import tempfile
    import numpy as _np

    samples = params.get("samples", 64)
    engine = params.get("engine", "CYCLES")
    denoise = params.get("denoise", True)

    _configure_render(engine, samples, denoise)

    w = bpy.context.scene.render.resolution_x
    h = bpy.context.scene.render.resolution_y

    # Render to a temporary file, then read pixels back.
    # Blender's Render Result image doesn't expose pixels reliably
    # without write_still, and going through a temp file with
    # OpenEXR/PNG is the most reliable path.
    tmp_path = os.path.join(tempfile.gettempdir(), f"_robo_blender_render_{os.getpid()}.png")
    bpy.context.scene.render.filepath = tmp_path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.ops.render.render(write_still=True)

    # Read the PNG back as raw pixels
    img = bpy.data.images.load(tmp_path)
    pixels_float = list(img.pixels)
    bpy.data.images.remove(img)

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    n = w * h * 4
    if len(pixels_float) < n:
        raise RuntimeError(f"Pixel buffer too small: {len(pixels_float)} < {n}")

    # Convert to uint8, flip vertically (Blender is bottom-up)
    arr = _np.array(pixels_float[:n], dtype=_np.float32).reshape(h, w, 4)
    arr = _np.flipud(arr)
    arr_u8 = (_np.clip(arr, 0.0, 1.0) * 255).astype(_np.uint8)

    raw_bytes = arr_u8.tobytes()
    b64 = base64.b64encode(raw_bytes).decode("ascii")

    return {
        "width": w,
        "height": h,
        "channels": 4,
        "encoding": "base64_rgba_uint8",
        "data": b64,
    }


def _normalize_engine(engine: str) -> str:
    """Normalize engine name for compatibility across Blender versions.

    Blender 4.x uses BLENDER_EEVEE_NEXT, Blender 5.x uses BLENDER_EEVEE.
    Accept both plus short aliases.
    """
    eng = engine.upper()
    if eng in ("EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        # Check which name this Blender version accepts
        if bpy.app.version >= (5, 0, 0):
            return "BLENDER_EEVEE"
        else:
            return "BLENDER_EEVEE_NEXT"
    return eng


def _configure_render(engine: str, samples: int, denoise: bool,
                       hero_quality: bool = False,
                       film_transparent: bool = False,
                       color_depth: str = "8"):
    """Common render configuration for both file and bytes rendering.

    hero_quality (Cycles only): sharper filter, accurate denoiser prefilter,
    RGB+albedo+normal input passes, disables render.use_simplify, DOF, and
    motion blur.  Intended for paper-figure hero shots.
    """
    scene = bpy.context.scene
    engine = _normalize_engine(engine)
    scene.render.engine = engine

    if engine == "CYCLES":
        _setup_gpu()
        scene.cycles.device = "GPU"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = denoise
        if denoise:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.005 if hero_quality else 0.01
        if hero_quality:
            # Nothing silently downsamples textures / meshes.
            scene.render.use_simplify = False
            # Sharper pixel filter (default 1.5 = noticeably soft edges).
            try:
                scene.cycles.filter_width = 1.0
            except Exception:
                pass
            # Kill intentional blur.
            try:
                if scene.camera is not None and scene.camera.data is not None:
                    scene.camera.data.dof.use_dof = False
            except Exception:
                pass
            scene.render.use_motion_blur = False
            # High-fidelity denoiser input.
            try:
                scene.cycles.denoising_prefilter = "ACCURATE"
                scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
                scene.cycles.denoising_use_gpu = True
            except Exception:
                pass
    else:
        # ── EEVEE quality tuning for softer, more Cycles-like results ──
        eevee = scene.eevee

        # Shadows: more rays + steps = softer penumbra, less blocky edges.
        # Default is 1 ray / 6 steps — bumping to 3/16 gives much softer
        # shadows with only ~20-30% speed cost.
        eevee.shadow_ray_count = 3
        eevee.shadow_step_count = 16
        eevee.shadow_resolution_scale = 1.0  # full-res shadow maps

        # Jitter shadow samples to reduce banding in soft shadow penumbra
        eevee.use_shadow_jitter_viewport = True

        # Temporal anti-aliasing: higher sample count for final render
        # smooths edges and reduces noise.
        eevee.taa_render_samples = samples

        # Global illumination (Fast GI): enables indirect light bounces
        # for softer ambient lighting in shadowed areas.
        eevee.use_fast_gi = True
        eevee.fast_gi_method = "GLOBAL_ILLUMINATION"
        eevee.fast_gi_ray_count = 2
        eevee.fast_gi_quality = 0.5       # higher = less noise (0.25 default)
        eevee.fast_gi_step_count = 8
        eevee.fast_gi_resolution = '2'       # 1/2 res — good speed/quality

        # Screen-space ray tracing for reflections (subtle, but helps
        # metallic/glossy surfaces look more natural).
        eevee.use_raytracing = True
        eevee.ray_tracing_method = "SCREEN"
        rt = eevee.ray_tracing_options
        rt.resolution_scale = '2'            # 1/2 res traces — fast
        rt.screen_trace_quality = 0.5      # balance speed vs quality
        rt.use_denoise = True
        rt.denoise_spatial = True
        rt.denoise_temporal = True
        rt.denoise_bilateral = True

        # Use high-quality normals for smoother shading on curved surfaces
        scene.render.use_high_quality_normals = True

        # Slightly larger pixel filter for anti-aliased edges
        scene.render.filter_size = 1.5

    # Color management
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0

    # Output format tweaks (hero-quality only).
    if hero_quality:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        try:
            scene.render.image_settings.color_depth = str(color_depth)
        except Exception:
            pass
        scene.render.film_transparent = bool(film_transparent)


def insert_keyframes(params: dict) -> dict:
    """Insert animation keyframes for all registered bodies.

    Expects:
        keyframes: list of {frame: int, transforms: [{body_id, link_index,
                   pos, orn_xyzw}, ...]}
        fps:       playback frame rate (sets scene.render.fps)

    Sets scene.frame_start, scene.frame_end, scene.render.fps from the data.
    """
    keyframes = params.get("keyframes", [])
    fps = params.get("fps", 30)

    if not keyframes:
        return {"inserted": 0}

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1

    max_frame = 0
    inserted = 0

    for snapshot in keyframes:
        frame = snapshot["frame"]
        if frame > max_frame:
            max_frame = frame

        for t in snapshot["transforms"]:
            key = _key(t["body_id"], t["link_index"])
            obj_name = _registry.get(key)
            if not obj_name or obj_name not in bpy.data.objects:
                continue

            obj = bpy.data.objects[obj_name]
            obj.location = mathutils.Vector(t["pos"])
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = _quat_pb2bl(t["orn_xyzw"])

            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            inserted += 1

    scene.frame_end = max_frame

    # Set all animation curves to linear interpolation (no ease-in/out)
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            try:
                for fcurve in obj.animation_data.action.fcurves:
                    for kf in fcurve.keyframe_points:
                        kf.interpolation = 'LINEAR'
            except Exception as e:
                print(f"[insert_keyframes] Warning: could not set linear "
                      f"interpolation on {obj.name}: {e}")

    print(f"[insert_keyframes] {inserted} keyframes across {len(keyframes)} "
          f"frames (1..{max_frame}), fps={fps}")
    return {"inserted": inserted, "total_frames": max_frame, "fps": fps}


def render_animation(params: dict) -> dict:
    """Render the animated timeline to an MP4 video.

    Renders PNG frames via Blender, then stitches to H.264 MP4 with ffmpeg.
    This avoids the FFMPEG enum issue in Blender 5.x builds where the
    ``FFMPEG`` file_format is not available despite being in the schema.

    Frame range and FPS must already be set (by insert_keyframes).
    """
    import subprocess, tempfile, glob as _glob, shutil

    output_path = os.path.abspath(params["output_path"])
    samples = params.get("samples", 16)
    engine = params.get("engine", "EEVEE")
    frame_step = params.get("frame_step", 3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    _configure_render(engine, samples, denoise=False)

    scene = bpy.context.scene
    fps = scene.render.fps
    start = scene.frame_start
    end = scene.frame_end

    # Render to PNG frames in a temp directory
    tmpdir = tempfile.mkdtemp(prefix="blend_anim_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = os.path.join(tmpdir, "frame_")
    scene.frame_step = frame_step

    output_fps = max(1, fps // frame_step)
    n_to_render = len(range(start, end + 1, frame_step))

    print(f"[render_animation] Rendering {n_to_render} frames "
          f"(step={frame_step}) @ {output_fps}fps → {output_path}")

    bpy.ops.render.render(animation=True)

    # Stitch with ffmpeg
    frames = sorted(_glob.glob(os.path.join(tmpdir, "frame_*.png")))
    if not frames:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {"error": "No frames rendered", "output_path": output_path}

    print(f"[render_animation] Stitching {len(frames)} frames with ffmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(output_fps),
        "-pattern_type", "glob",
        "-i", os.path.join(tmpdir, "frame_*.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    if result.returncode != 0:
        err = result.stderr.decode()[-300:]
        print(f"[render_animation] ffmpeg error: {err}")
        return {"error": f"ffmpeg failed: {err}", "output_path": output_path}

    print(f"[render_animation] Done: {output_path}")
    return {"output_path": output_path,
            "frames": len(frames),
            "fps": output_fps}


def set_object_texture(params: dict) -> dict:
    """Repoint an already-loaded object's image texture to a new file.

    Used by the marked-comparison question type: a label badge is baked into a
    copy of the asset's texture map and applied at runtime.  PyBullet can do
    this with ``changeVisualShape(textureUniqueId=...)``, but Blender imported
    the OBJ with the ORIGINAL texture, so without this the VLM-facing render
    would show unbadged objects while the simulator shows badged ones.

    params:
        body_id, link_index: identify the body (as in load_body)
        texture_path:        absolute path to the new image
    """
    body_id = int(params["body_id"])
    link_index = int(params.get("link_index", -1))
    tex_path = params["texture_path"]

    if not os.path.isfile(tex_path):
        return {"error": f"texture not found: {tex_path}"}

    key = _key(body_id, link_index)
    obj_name = _registry.get(key)
    if obj_name is None:
        # Fall back to any link of this body (single-link assets are common).
        for (bid, lidx), nm in _registry.items():
            if bid == body_id:
                obj_name, key = nm, (bid, lidx)
                break
    if obj_name is None:
        return {"error": f"body {body_id} not in scene registry"}

    obj = bpy.data.objects.get(obj_name)
    if obj is None or not getattr(obj, "data", None):
        return {"error": f"object {obj_name} missing"}

    img = bpy.data.images.load(tex_path, check_existing=True)
    n_set = 0
    for mat in (obj.data.materials or []):
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                node.image = img
                n_set += 1
    if n_set == 0:
        # No image node yet (flat-coloured material): build a minimal one so
        # the badge still shows.
        for mat in (obj.data.materials or []):
            if not mat:
                continue
            mat.use_nodes = True
            nt = mat.node_tree
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
                n_set += 1
    return {"object": obj_name, "nodes_updated": n_set,
            "texture": os.path.basename(tex_path)}
