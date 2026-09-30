# RoboBlender — PyBullet ↔ Blender Socket Bridge
# See note.md for full architecture and design decisions.


def __getattr__(name: str):
    """Lazy import to avoid pulling pybullet into Blender's Python."""
    if name == "BlenderRenderer":
        from robo_blender.renderer import BlenderRenderer
        return BlenderRenderer
    if name == "BlenderProcess":
        from robo_blender.server_launcher import BlenderProcess
        return BlenderProcess
    if name == "download_random_hdri":
        from robo_blender.renderer import download_random_hdri
        return download_random_hdri
    raise AttributeError(f"module 'robo_blender' has no attribute {name!r}")


__all__ = ["BlenderRenderer", "BlenderProcess", "download_random_hdri"]
