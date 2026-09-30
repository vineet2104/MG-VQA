"""
Server Launcher — Auto-start Blender bridge server as a background process.

Finds a Blender binary, spawns it in background mode with the bridge server
script, waits for TCP readiness, and manages lifecycle (cleanup on exit).
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time


def _find_blender() -> str:
    """Find a Blender binary.

    Search order:
        1. BLENDER env var
        2. ``blender`` on PATH
        3. Common install locations (Linux, macOS)

    Returns:
        Absolute path to the blender executable.

    Raises:
        FileNotFoundError: If no Blender binary is found.
    """
    # 1. Explicit env var
    env_blender = os.environ.get("BLENDER", "")
    if env_blender:
        candidate = shutil.which(os.path.expanduser(env_blender))
        if not candidate:
            raise FileNotFoundError(f"BLENDER is not an executable: {env_blender}")
        return os.path.abspath(candidate)

    on_path = shutil.which("blender")
    if on_path:
        return os.path.abspath(on_path)

    # 2. Common locations — checked BEFORE PATH on purpose: a stale system
    #    Blender (e.g. /usr/bin/blender 4.x installed via apt) shadows the
    #    README install at ~/blender/blender, and older Blender builds ship
    #    a bundled Python without numpy — so every bridge render fails with
    #    "ModuleNotFoundError: No module named 'numpy'".
    candidates = []
    home = os.path.expanduser("~")

    # $HOME/blender/blender (README install path)
    readme_path = os.path.join(home, "blender", "blender")
    if os.path.isfile(readme_path):
        candidates.append(readme_path)

    # $HOME/blender-X.Y.Z-linux-x64/blender (download extract path)
    for entry in os.listdir(home) if os.path.isdir(home) else []:
        if entry.startswith("blender-") and os.path.isfile(
            os.path.join(home, entry, "blender")
        ):
            candidates.append(os.path.join(home, entry, "blender"))

    # Prefer highest version (lexicographic sort works for blender-X.Y.Z)
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0]

    # 3. On PATH (fallback — may be an older/system Blender)
    on_path = shutil.which("blender")
    if on_path:
        return os.path.abspath(on_path)

    # macOS Application bundle
    mac_path = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.isfile(mac_path):
        return mac_path

    raise FileNotFoundError(
        "Cannot find Blender. Set the BLENDER environment variable "
        "or add blender to your PATH."
    )


def _is_server_running(host: str, port: int) -> bool:
    """Check if something is already listening on host:port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect((host, port))
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    """Block until the server accepts TCP connections, or timeout.

    Returns:
        True if server became reachable, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_server_running(host, port):
            return True
        time.sleep(0.5)
    return False


class BlenderProcess:
    """Manages a Blender background process running the bridge server.

    The process is started with ``blender --background --python run_server.py``
    and is terminated when this object is garbage-collected, when ``stop()``
    is called, or when the Python process exits (via atexit).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        blender_path: str | None = None,
        startup_timeout: float = 30.0,
        quiet: bool = False,
    ):
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.quiet = quiet
        self._process: subprocess.Popen | None = None
        self._owned = False  # True if we started the process
        self.blender_path = blender_path or _find_blender()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _kill_stale_server(self) -> None:
        """Find and kill any Blender process holding our port."""
        try:
            # Find PIDs listening on our port
            result = subprocess.run(
                ["fuser", f"{self.port}/tcp"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            for pid_str in pids:
                pid_str = pid_str.strip()
                if pid_str.isdigit():
                    pid = int(pid_str)
                    if not self.quiet:
                        print(f"[BlenderProcess] Killing stale PID {pid}")
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # fuser not available — try connecting and sending shutdown
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((self.host, self.port))
                s.close()
            except Exception:
                pass

    def start(self) -> None:
        """Start the Blender server, killing any stale server first.

        Always ensures a fresh server for this session. Previous orphaned
        servers on the same port are terminated before starting.
        """
        # Kill any stale server on our port before starting a fresh one
        if _is_server_running(self.host, self.port):
            if not self.quiet:
                print(
                    f"[BlenderProcess] Killing stale server on "
                    f"{self.host}:{self.port}..."
                )
            self._kill_stale_server()
            # Wait for port to free up
            for _ in range(10):
                if not _is_server_running(self.host, self.port):
                    break
                time.sleep(0.5)

        if self._process is not None:
            return

        server_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scripts",
            "run_server.py",
        )

        if not os.path.isfile(server_script):
            raise FileNotFoundError(
                f"Server script not found: {server_script}"
            )

        env = os.environ.copy()
        env["ROBO_BRIDGE_HOST"] = self.host
        env["ROBO_BRIDGE_PORT"] = str(self.port)
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # prevent stale .pyc in Blender's Python
        # Note: Blender EEVEE needs NVIDIA EGL libs for headless GPU rendering.
        # The caller must set LD_LIBRARY_PATH to include the CUDA/EGL libs
        # (e.g. /cm/local/apps/cuda/libs/current/lib64 on compute nodes).
        # Do NOT force --gpu-backend egl — let Blender auto-detect.

        # Always log Blender output to file for debugging crashes
        self._log_path = os.path.join(
            tempfile.gettempdir(), f"blender_server_{self.port}.log"
        )
        self._log_file = open(self._log_path, "w")

        if not self.quiet:
            print(
                f"[BlenderProcess] Starting Blender server: "
                f"{self.blender_path} --background --python {server_script}"
            )
        print(f"[BlenderProcess] Server log: {self._log_path}")

        self._process = subprocess.Popen(
            [self.blender_path, "--background", "--python", server_script],
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            # Start in its own process group so we can kill it cleanly
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
        self._owned = True

        # Register cleanup
        atexit.register(self.stop)

        # Wait for TCP readiness
        if not _wait_for_server(self.host, self.port, self.startup_timeout):
            self.stop()
            raise TimeoutError(
                f"Blender server did not start within {self.startup_timeout}s"
            )

        if not self.quiet:
            print(
                f"[BlenderProcess] Server ready on {self.host}:{self.port} "
                f"(PID {self._process.pid})"
            )

    def stop(self) -> None:
        """Terminate the Blender process if we started it."""
        # Close the log file
        if hasattr(self, "_log_file") and self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        if self._process is None or not self._owned:
            return

        proc = self._process
        self._process = None

        if proc.poll() is not None:
            return

        try:
            # Kill the entire process group
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except (OSError, ProcessLookupError):
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (OSError, ProcessLookupError):
                pass

    def __del__(self):
        self.stop()
