"""
Bridge Server — TCP server running inside Blender (--background mode).

- Listener thread accepts connections, spawns client handlers.
- Client handlers push commands to a queue.
- Main thread drains queue and executes bpy operations.
"""

import bpy
import socket
import threading
import queue
import traceback
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_this_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from robo_blender.protocol.messages import (
    encode_message, decode_message,
    response_ok, response_error,
)
from robo_blender.server import scene_manager

# ---- State ----
_cmd_queue: queue.Queue = queue.Queue()
_server_socket: socket.socket | None = None
_shutdown = threading.Event()

# ---- Dispatch ----
HANDLERS = {
    "PING":                    lambda p: {"pong": True},
    "LOAD_BODY":               scene_manager.load_body,
    "UPDATE_TRANSFORM":        scene_manager.update_transform,
    "BATCH_UPDATE_TRANSFORMS": scene_manager.batch_update_transforms,
    "SET_CAMERA":              scene_manager.set_camera,
    "CONFIGURE_LIGHTING":      scene_manager.configure_lighting,
    "RENDER_FRAME":            scene_manager.render_frame,
    "RENDER_FRAME_BYTES":      scene_manager.render_frame_bytes,
    "SAVE_BLEND":              scene_manager.save_blend,
    "SET_TABLE_TEXTURE":       scene_manager.set_table_texture,
    "SET_OBJECT_TEXTURE":      scene_manager.set_object_texture,
    "SET_WORLD_HDRI":          scene_manager.set_world_hdri,
    "INSERT_KEYFRAMES":        scene_manager.insert_keyframes,
    "RENDER_ANIMATION":        scene_manager.render_animation,
    "REMOVE_BODIES":           scene_manager.remove_bodies,
    "CLEAR_SCENE":             lambda p: (scene_manager.clear_scene(), {"cleared": True})[1],
    "SHUTDOWN":                lambda p: {"shutdown": True},
}


def _handle(msg: dict) -> dict:
    cmd = msg.get("cmd", "")
    msg_id = msg.get("id", "?")
    params = msg.get("params", {})
    handler = HANDLERS.get(cmd)
    if not handler:
        return response_error(msg_id, f"Unknown command: {cmd}")
    try:
        return response_ok(msg_id, **handler(params))
    except Exception as e:
        print(f"[Server] ERROR {cmd}: {e}\n{traceback.format_exc()}")
        return response_error(msg_id, str(e))


def _client_thread(conn: socket.socket, addr):
    print(f"[Server] Client connected: {addr}")
    try:
        while not _shutdown.is_set():
            try:
                msg = decode_message(conn)
            except ConnectionError:
                break
            except Exception as e:
                print(f"[Server] Decode error: {e}")
                break

            resp_q = queue.Queue()
            _cmd_queue.put((msg, resp_q))

            try:
                response = resp_q.get(timeout=600)
            except queue.Empty:
                response = response_error(msg.get("id", "?"), "Timeout")

            try:
                conn.sendall(encode_message(response))
            except Exception:
                break

            if msg.get("cmd") == "SHUTDOWN":
                break
    finally:
        conn.close()
        print(f"[Server] Client disconnected: {addr}")


def _listener(host: str, port: int):
    global _server_socket
    _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _server_socket.settimeout(1.0)
    _server_socket.bind((host, port))
    _server_socket.listen(2)
    print(f"[Server] Listening on {host}:{port}")

    while not _shutdown.is_set():
        try:
            conn, addr = _server_socket.accept()
            conn.settimeout(None)
            threading.Thread(target=_client_thread, args=(conn, addr), daemon=True).start()
        except socket.timeout:
            continue
        except OSError:
            break


def start_server(host: str = "127.0.0.1", port: int = 9876):
    """Start server. Blocks until SHUTDOWN received."""
    _shutdown.clear()
    scene_manager.clear_scene()

    threading.Thread(target=_listener, args=(host, port), daemon=True).start()
    print(f"[Server] Ready on {host}:{port}")

    # Main-thread loop: drain queue, execute bpy ops
    while not _shutdown.is_set():
        try:
            msg, resp_q = _cmd_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        resp = _handle(msg)
        resp_q.put(resp)

        if msg.get("cmd") == "SHUTDOWN":
            _shutdown.set()
            break

    # Cleanup
    if _server_socket:
        try:
            _server_socket.close()
        except Exception:
            pass
    print("[Server] Stopped.")
