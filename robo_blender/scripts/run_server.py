"""
Blender Bridge Server — Entry point.

    blender --background --python scripts/run_server.py
"""
import sys, os

# Go up two levels: scripts/ -> robo_blender/ -> repo root
# This puts the repo root on sys.path so `import robo_blender` works
# whether run_server.py lives at <repo>/robo_blender/scripts/ or <repo>/scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(script_dir)           # robo_blender/
repo_root = os.path.dirname(package_dir)             # Agentic_MQA/

for p in (repo_root, package_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from robo_blender.server.bridge_server import start_server

HOST = os.environ.get("ROBO_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("ROBO_BRIDGE_PORT", "9876"))

print(f"[run_server] {HOST}:{PORT}  repo_root={repo_root}")
start_server(host=HOST, port=PORT)
