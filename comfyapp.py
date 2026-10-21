import subprocess
import sys
import time
import json
import uuid
import os
import modal

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

try:
    from workflow_inputs import stage_remote_input_images
except ImportError:
    # Volume-helper functions (sync/status/upload) run on the slim image,
    # which intentionally has no local sources. They never call this helper
    # (only run_prompt does, and it runs on the main image which has it).
    # A hard top-level import would crash-loop those functions at import.
    def stage_remote_input_images(*args, **kwargs):
        raise RuntimeError("workflow_inputs module not available in this image")

# Bump this version whenever comfyapp.py changes.
# The custom node compares this against the last deployed version
# and re-runs `modal deploy` only when the version changes.
#
# SECURITY: commenting out `ui()` below removes the app's only public
# endpoint, but it only takes effect after a redeploy. Press **Deploy to
# Cloud** in the node to apply it (the fork removed the startup auto-deploy,
# so bumping this version alone will NOT trigger a redeploy). Bumping it
# (e.g. to "2.0.10") is still good practice so the tracked version matches
# the code after the next deploy. The node itself never uses that URL (it
# uses the SDK + your Modal API token), so removing it cannot break Deploy /
# Check / Wake Up / prompts / model management.
COMFYAPP_VERSION = "2.0.12"

# ComfyUI release installed into the Modal image.
#
# WHY THIS HAS TO BE PINNED (why the container ran 0.38.0 while local ran 0.42.1):
# `comfy install` defaults to `--version nightly`, i.e. "whatever the latest
# commit on ComfyUI's master branch was at BUILD time". Modal caches the
# image layer and `nightly` is never re-resolved, so the container stayed
# frozen on the nightly the layer was built with (v0.38.0-8-g8cfe5e1e) even
# though the local ComfyUI was already 0.42.1 -> workflows failed validation
# with "KSampler: Exception when validating inner node: tuple index out of range".
#
# Pinning an explicit release does two things:
#   1. installs the exact same version as the local ComfyUI, so workflows
#      validate identically;
#   2. changes the build command, so Modal's image-layer cache misses and the
#      image is genuinely rebuilt instead of silently reusing the old one.
#
# To update ComfyUI on Modal: change this number and redeploy. Every change
# to it forces a fresh image build.
COMFYUI_VERSION = "0.42.1"

APP_NAME = "comfyui"
VOLUME_NAME = "comfyui-models"
CUSTOM_NODES_VOLUME_NAME = "comfyui-custom-nodes"
COMFYUI_PORT = 8188
COMFYUI_API_PORT = 8189
MODELS_PATH = "/root/models"
CUSTOM_NODES_PATH = "/root/custom_nodes_vol"

SUPPORTED_GPUS = ["a10g", "a100", "t4"]

# NOTE: Do not use `.apt_install()` in separate layers on the legacy image
# builder: its default `debian_slim` base is bullseye, whose
# bullseye-security APT pool now returns 404 for many packages
# (openssh-client, libasound2, ffmpeg, mesa, ...).
# Pin bookworm explicitly and run `apt-get update + install` in a single
# RUN layer so the package index can never go stale.
# NOTE: gcc is required at RUNTIME, not just build time: torch's Triton
# kernels (e.g. bmm_outer_product used by newer text encoders like
# MiniMax/Qwen3-VL) JIT-compile a small CUDA helper on first use, and
# Triton aborts with "Failed to find C compiler" when cc is missing.
# libc6-dev is needed alongside it: we install with --no-install-recommends,
# which would otherwise skip the C standard headers (stdlib.h, ...).
APT_PACKAGES = "git gcc libc6-dev libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg"


def _base_image():
    """Bookworm-based Python image, working on both legacy and new builders."""
    try:
        from_registry = getattr(modal.Image, "from_registry", None)
        if callable(from_registry):
            return from_registry("python:3.11-slim-bookworm")
    except Exception:
        pass
    return modal.Image.debian_slim(python_version="3.11")


def _apt_install_cmd():
    return (
        "apt-get update && "
        f"DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {APT_PACKAGES} "
        "&& rm -rf /var/lib/apt/lists/*"
    )


image = (
    _base_image()
    .run_commands(_apt_install_cmd())
    .pip_install("comfy-cli==1.23.0")
    .run_commands(
        # --version pins the exact ComfyUI release (see COMFYUI_VERSION above).
        # Without it `comfy install` falls back to `--version nightly`, which
        # freezes the image on whatever master was at build time (0.38.0)
        # forever, because Modal caches the layer and never re-resolves it.
        # --workspace pins the install location so a comfy-cli default change
        # can never move ComfyUI out from under the hardcoded /root/comfy/ComfyUI
        # paths used by the model/custom-node/input helpers below.
        f"comfy --skip-prompt --workspace /root/comfy/ComfyUI install --nvidia --version {COMFYUI_VERSION}",
        gpu="a10g",
    )
    .add_local_python_source("workflow_inputs")
)

# Shared slim base for all lightweight volume functions.
# IMPORTANT: use ONE object so Modal builds/caches ONE image.
# (Calling _base_image() per-function creates distinct images and forces
# a separate remote build on first call of each function.)
_slim_image = _base_image()

download_image = (
    _slim_image
    .pip_install("httpx>=0.27.0")
    .add_local_python_source("workflow_inputs")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
custom_nodes_vol = modal.Volume.from_name(CUSTOM_NODES_VOLUME_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# SECURITY: the public ComfyUI web UI is DISABLED here.
#
# `ui()` is a Modal Web Function, and Modal Web Functions are PUBLIC by
# default unless `requires_proxy_auth=True` is set. With the plain decorator
# below, anyone who learns the deterministic URL
# (https://<workspace>--comfyui-ui.modal.run) gets a full ComfyUI UI + API on
# an a10g container, with the models and custom-nodes volumes mounted.
# See investigation/02-cbsakura-security-assessment.md.
#
# The comfyui-modal node does NOT use this endpoint: it talks to the app
# through the Modal SDK (.remote()), authenticated by the Modal API token
# (ak-/as-) in ~/.modal.toml — not by this web URL. So disabling `ui()` does
# not affect the node, Deploy, Check, Wake Up, prompts, or model management.
#
# To re-enable it SAFELY, uncomment the block below AND add
# `requires_proxy_auth=True` to the `@modal.web_server(...)` call. Then create
# a Modal Proxy Token (Settings -> Proxy Auth Tokens; id `wk-`, secret `ws-`)
# and send it as `Modal-Key`/`Modal-Secret` headers, or as
# `Authorization: Bearer wk-...ws-...`. See investigation/03-modal-proxy-auth-howto.md.
# ---------------------------------------------------------------------------
# @app.function(
#     gpu="a10g",
#     cpu=4,
#     memory=16384,
#     timeout=3600,
#     min_containers=0,
#     scaledown_window=2,
#     volumes={MODELS_PATH: vol, CUSTOM_NODES_PATH: custom_nodes_vol},
# )
# @modal.web_server(COMFYUI_PORT, startup_timeout=300)
# def ui():
#     subprocess.Popen(
#         f"comfy launch -- --listen 0.0.0.0 --port {COMFYUI_PORT}",
#         shell=True,
#     )
#
# NOTE: COMFYUI_PORT (8188) is only used by the disabled ui() above; it is
# left defined so re-enabling the block is a one-step change.


@app.function(
    image=download_image,
    cpu=2,
    memory=512,
    timeout=1800,
    volumes={MODELS_PATH: vol},
)
def download_model_to_volume(url: str, filename: str, save_path: str = "checkpoints", hf_token: str = ""):
    import httpx
    from pathlib import Path

    dest = Path(MODELS_PATH) / save_path / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return {"status": "ok", "skipped": True, "path": str(dest)}

    headers = {}
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=1800) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1048576):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    sys.stdout.write(f"\r  {pct:.1f}%  ({downloaded // 1024**2} MB / {total // 1024**2} MB)")
                    sys.stdout.flush()

    vol.commit()
    return {"status": "ok", "path": str(dest)}


@app.function(
    image=download_image,
    cpu=2,
    memory=512,
    timeout=1800,
    volumes={MODELS_PATH: vol},
)
def batch_download_models(items: list, hf_token: str = "") -> list:
    results = list(
        download_model_to_volume.starmap(
            [(item["url"], item["filename"], item.get("save_path", "checkpoints"), hf_token) for item in items]
        )
    )
    return results


@app.function(
    image=_slim_image,
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes={CUSTOM_NODES_PATH: custom_nodes_vol},
)
def sync_custom_nodes_to_volume(archive_data: bytes) -> dict:
    """Receive a tar.gz archive of custom nodes and extract to volume."""
    import tarfile
    import io
    import os
    import shutil

    staging_dir = os.path.join(CUSTOM_NODES_PATH, ".staging")

    # Clean any leftover staging dir
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    # Extract to staging with path traversal protection
    buf = io.BytesIO(archive_data)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            # Reject absolute paths and parent references
            if member.name.startswith("/") or ".." in member.name.split("/"):
                raise ValueError(f"Tar member '{member.name}' contains unsafe path")
            # Verify resolved path stays within staging directory
            member_path = os.path.normpath(os.path.join(staging_dir, member.name))
            if not member_path.startswith(os.path.normpath(staging_dir)):
                raise ValueError(f"Tar member '{member.name}' would extract outside target directory")
        # Reset buffer and extract after validation
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar2:
            tar2.extractall(path=staging_dir)

    # Swap: remove old content, move staging content into place
    for item in os.listdir(CUSTOM_NODES_PATH):
        if item == ".staging":
            continue
        item_path = os.path.join(CUSTOM_NODES_PATH, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

    # Move extracted items from staging to volume root
    for item in os.listdir(staging_dir):
        src = os.path.join(staging_dir, item)
        dst = os.path.join(CUSTOM_NODES_PATH, item)
        shutil.move(src, dst)

    # Clean up staging
    shutil.rmtree(staging_dir)

    custom_nodes_vol.commit()

    # List what was extracted
    nodes = [d for d in os.listdir(CUSTOM_NODES_PATH) if os.path.isdir(os.path.join(CUSTOM_NODES_PATH, d))]
    print(f"[comfyui-modal] Synced {len(nodes)} custom nodes ({len(archive_data) / 1024**2:.1f} MB): {sorted(nodes)}")
    return {"status": "ok", "nodes": sorted(nodes)}


@app.function(
    image=_slim_image,
    cpu=1,
    memory=512,
    timeout=60,
    volumes={MODELS_PATH: vol, CUSTOM_NODES_PATH: custom_nodes_vol},
)
def get_volume_status() -> dict:
    """Return current state of both volumes."""
    import os

    vol.reload()
    custom_nodes_vol.reload()

    # Scan models
    models = []
    if os.path.isdir(MODELS_PATH):
        for folder in os.listdir(MODELS_PATH):
            folder_path = os.path.join(MODELS_PATH, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    models.append({"folder": folder, "name": fname, "size": os.path.getsize(fpath)})

    # Scan custom nodes
    custom_nodes = []
    if os.path.isdir(CUSTOM_NODES_PATH):
        for d in os.listdir(CUSTOM_NODES_PATH):
            if os.path.isdir(os.path.join(CUSTOM_NODES_PATH, d)):
                custom_nodes.append(d)

    return {"models": models, "custom_nodes": custom_nodes}


@app.function(
    image=_slim_image,
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes={MODELS_PATH: vol},
)
def upload_model_to_volume(file_data: bytes, folder: str, filename: str) -> dict:
    """Upload a model file directly to the volume."""
    import os
    from pathlib import Path

    dest = Path(MODELS_PATH) / folder / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        f.write(file_data)

    vol.commit()
    return {"status": "ok", "path": str(dest), "size": len(file_data)}


@app.function(
    image=_slim_image,
    cpu=2,
    memory=4096,
    timeout=3600,
    volumes={MODELS_PATH: vol},
)
def upload_model_chunk(chunk_data: bytes, folder: str, filename: str, offset: int, is_last: bool) -> dict:
    """Upload a model file chunk to the volume. Chunks are appended sequentially."""
    import os
    from pathlib import Path

    dest = Path(MODELS_PATH) / folder / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    mode = "ab" if offset > 0 else "wb"
    with open(dest, mode) as f:
        f.write(chunk_data)

    if is_last:
        vol.commit()
        return {"status": "ok", "path": str(dest), "size": os.path.getsize(dest)}

    return {"status": "partial", "offset": offset + len(chunk_data)}


class _ComfyAPIMixin:
    """Shared implementation for all GPU-specific ComfyAPI classes."""

    @modal.enter(snap=True)
    def startup(self):
        import os
        import shutil

        # Symlink models from volume into ComfyUI
        comfy_models = "/root/comfy/ComfyUI/models"
        if os.path.isdir(comfy_models):
            shutil.rmtree(comfy_models)
        os.symlink(MODELS_PATH, comfy_models)

        self._link_custom_nodes_from_volume()
        self._start_comfy()

    @modal.enter(snap=False)
    def restore(self):
        # A sync may have added new custom nodes to the volume AFTER this
        # container was snapshotted. Link anything new and restart ComfyUI
        # so the new node types are actually loaded (ComfyUI only scans
        # custom_nodes at process start).
        try:
            new_nodes = self._link_custom_nodes_from_volume()
        except Exception as e:
            print(f"[comfyui-modal] WARNING: custom-node relink failed: {e}")
            new_nodes = []
        if new_nodes:
            print(f"[comfyui-modal] New custom nodes detected ({new_nodes}), restarting ComfyUI...")
            self._restart_comfy()
        else:
            self._wait_for_comfy()

    def _link_custom_nodes_from_volume(self) -> list:
        """Symlink volume custom nodes into ComfyUI. Returns newly linked dirs."""
        import os

        vol.reload()
        custom_nodes_vol.reload()
        comfy_custom_nodes = "/root/comfy/ComfyUI/custom_nodes"
        vol_cn_path = CUSTOM_NODES_PATH
        newly_linked = []
        req_files = []
        if os.path.isdir(vol_cn_path):
            for node_dir in sorted(os.listdir(vol_cn_path)):
                src = os.path.join(vol_cn_path, node_dir)
                dst = os.path.join(comfy_custom_nodes, node_dir)
                if os.path.isdir(src) and not os.path.exists(dst):
                    os.symlink(src, dst)
                    newly_linked.append(node_dir)
                    # Install requirements if present.
                    # NOTE: ComfyUI never imports directories whose name ends
                    # in ".disabled" (nodes.py: `if module_path.endswith(".disabled"):
                    # continue`), so their dependencies are never imported and
                    # installing them is pure waste on every container start.
                    if node_dir.endswith(".disabled"):
                        continue
                    req_file = os.path.join(src, "requirements.txt")
                    if os.path.isfile(req_file):
                        req_files.append((node_dir, req_file))
        if newly_linked:
            print(f"[comfyui-modal] Linked custom nodes: {newly_linked}")
        if req_files:
            # NOTE: install every new node's requirements in ONE pip call so
            # the dependency resolution (the slow part: index round-trips) is
            # shared instead of paid once per node. Installing them one by one
            # cost ~2 minutes for the synced set, and because the memory
            # snapshot predates the sync that cost was repeated on EVERY
            # container start. If the combined install fails, fall back to
            # per-node installs so a single bad node can be identified.
            print(f"[comfyui-modal] Installing requirements for: {[n for n, _ in req_files]}")
            cmd = [sys.executable, "-m", "pip", "install"]
            for _, req_file in req_files:
                cmd += ["-r", req_file]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode != 0:
                    print(
                        "[comfyui-modal] WARNING: combined requirements install failed, "
                        "retrying per node to isolate the culprit: "
                        f"{(result.stderr or result.stdout)[-2000:]}"
                    )
                    for node_dir, req_file in req_files:
                        try:
                            single = subprocess.run(
                                [sys.executable, "-m", "pip", "install", "-r", req_file],
                                capture_output=True, text=True, timeout=600,
                            )
                            if single.returncode != 0:
                                print(
                                    f"[comfyui-modal] WARNING: requirements failed for {node_dir}: "
                                    f"{(single.stderr or single.stdout)[-2000:]}"
                                )
                        except Exception as e:
                            print(f"[comfyui-modal] WARNING: Failed to install requirements for {node_dir}: {e}")
            except Exception as e:
                print(f"[comfyui-modal] WARNING: Failed to install requirements: {e}")
        return newly_linked

    def _start_comfy(self):
        self._proc = subprocess.Popen(
            ["comfy", "launch", "--", "--listen", "0.0.0.0",
             f"--port={COMFYUI_API_PORT}", "--disable-auto-launch"],
        )
        self._wait_for_comfy()

    def _restart_comfy(self):
        proc = getattr(self, "_proc", None)
        # NOTE: guard the terminate calls. On the memory-snapshot restore path
        # `self._proc` refers to a process that belongs to the SNAPSHOT, not to
        # this container, so terminate()/wait() can raise (e.g. ProcessLookupError
        # / ChildProcessError). An exception here would abort the enter handler
        # and kill the container, which is what turned one bad restart into an
        # endless container churn.
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception as e:
                print(f"[comfyui-modal] NOTE: could not stop the previous ComfyUI process: {e}")
        # NOTE: on the memory-snapshot restore path `self._proc` is a Popen
        # object restored from the snapshot. terminate() does NOT reliably stop
        # the RESTORED ComfyUI process, which keeps port 8189 bound. The
        # replacement `comfy launch` then dies with
        # "Port 8189 is already in use on address 0.0.0.0" and the container
        # silently keeps serving the STALE ComfyUI from the snapshot (i.e. the
        # OLD custom-node versions) -> workflows fail validation with
        # "Exception when validating inner node: tuple index out of range".
        # Kill the real ComfyUI processes and wait until the port is actually
        # free before relaunching, otherwise the reload silently does nothing.
        self._kill_comfyui_processes()
        self._wait_for_port_free()
        self._start_comfy()

    def _kill_comfyui_processes(self):
        """SIGTERM (then SIGKILL) the running ComfyUI server process.

        `pkill` is not available in the slim image (no `procps` package), so
        /proc is scanned directly. Only a process whose command line mentions
        `main.py` AND whose working directory is the ComfyUI install dir is
        targeted, so the Modal worker can never be hit. The `comfy launch`
        wrapper exits by itself once its child is gone.
        """
        import signal

        comfy_dir = "/root/comfy/ComfyUI"

        def target_pids():
            pids = []
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == os.getpid():
                    continue
                try:
                    with open(f"/proc/{entry}/cmdline", "rb") as f:
                        cmdline = f.read().decode("utf-8", "replace").replace("\x00", " ")
                except Exception:
                    continue
                if "main.py" not in cmdline:
                    continue
                try:
                    cwd = os.readlink(f"/proc/{entry}/cwd")
                except Exception:
                    continue
                if cwd.rstrip("/").lower() != comfy_dir.lower():
                    continue
                pids.append(pid)
            return pids

        for sig in (signal.SIGTERM, signal.SIGKILL):
            pids = target_pids()
            if not pids:
                return
            print(f"[comfyui-modal] Sending {sig.name} to ComfyUI process(es): {pids}")
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except Exception:
                    pass
            for _ in range(50):
                if not target_pids():
                    return
                time.sleep(0.2)

    def _wait_for_port_free(self, timeout=60):
        """Block until nothing is listening on the ComfyUI API port.

        Starting a second ComfyUI while the first one still holds the port
        makes the new process exit immediately with "Port 8189 is already in
        use", which is exactly what silently broke the custom-node reload.
        """
        import socket

        deadline = time.time() + timeout
        while time.time() < deadline:
            sock = socket.socket()
            sock.settimeout(1)
            try:
                sock.connect(("127.0.0.1", COMFYUI_API_PORT))
            except OSError:
                return True
            finally:
                sock.close()
            time.sleep(0.5)
        print(f"[comfyui-modal] WARNING: port {COMFYUI_API_PORT} still busy after {timeout}s")
        return False

    @modal.exit()
    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _wait_for_comfy(self):
        import urllib.request
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_API_PORT}/system_stats")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("ComfyUI API failed to start")

    @modal.method()
    def object_info(self):
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_API_PORT}/object_info") as r:
            return json.loads(r.read())

    @modal.method()
    def run_prompt(self, workflow: dict, input_images: dict = None) -> dict:
        import urllib.request
        import urllib.error
        from pathlib import Path

        if input_images:
            stage_remote_input_images(
                Path("/root/comfy/ComfyUI/input"),
                input_images,
            )

        client_id = str(uuid.uuid4())
        payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{COMFYUI_API_PORT}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as r:
                queued = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Read the response body for validation error details
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
                error_data = json.loads(error_body)
                # Extract meaningful error info from ComfyUI's response
                node_errors = error_data.get("node_errors", {})
                if node_errors:
                    msgs = []
                    missing_types = []
                    for node_id, err_info in node_errors.items():
                        class_type = err_info.get("class_type", f"Node {node_id}")
                        for err in err_info.get("errors", []):
                            msgs.append(f"{class_type}: {err.get('message', 'unknown error')}")
                            if err.get("type") == "missing_node_type":
                                missing_types.append(class_type)
                    if msgs:
                        detail = '; '.join(msgs)
                        if missing_types:
                            detail += (
                                " | Missing custom node(s) on Modal cloud: "
                                + ", ".join(sorted(set(missing_types)))
                                + ". Run 'Sync Custom Nodes' in the Modal panel, "
                                + "wait ~60s for a cold start, then retry."
                            )
                        raise RuntimeError(f"Workflow validation failed: {detail}") from e
                # Fallback: use the message field
                msg = error_data.get("message", "") or error_data.get("error", "")
                if msg:
                    raise RuntimeError(f"Workflow validation failed: {msg}") from e
            except (json.JSONDecodeError, RuntimeError):
                if isinstance(sys.exc_info()[1], RuntimeError):
                    raise
            raise RuntimeError(f"ComfyUI rejected the prompt (HTTP {e.code}): {error_body[:500]}") from e

        prompt_id = queued["prompt_id"]
        return self._poll_until_done(prompt_id, client_id)

    def _poll_until_done(self, prompt_id: str, client_id: str) -> dict:
        import urllib.request
        delay = 0.5
        elapsed = 0.0
        while elapsed < 3600:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{COMFYUI_API_PORT}/history/{prompt_id}"
            ) as r:
                history = json.loads(r.read())
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                return self._collect_outputs(outputs)
            time.sleep(delay)
            elapsed += delay
            delay = min(delay * 1.5, 5.0)
        raise TimeoutError(f"Prompt {prompt_id} timed out")

    def _collect_outputs(self, outputs: dict) -> dict:
        import urllib.request
        import base64

        images = []
        videos = []

        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                animated = node_output.get("animated", (False,))
                is_animated = animated[0] if animated else False
                url = (
                    f"http://127.0.0.1:{COMFYUI_API_PORT}/view"
                    f"?filename={img['filename']}&subfolder={img.get('subfolder','')}&type={img.get('type','output')}"
                )
                with urllib.request.urlopen(url) as r:
                    data = base64.b64encode(r.read()).decode()
                entry = {"filename": img["filename"], "data": data, "node_id": node_id}
                if is_animated:
                    videos.append(entry)
                else:
                    images.append(entry)

            for vid in node_output.get("gifs", []):
                url = (
                    f"http://127.0.0.1:{COMFYUI_API_PORT}/view"
                    f"?filename={vid['filename']}&subfolder={vid.get('subfolder','')}&type={vid.get('type','output')}"
                )
                with urllib.request.urlopen(url) as r:
                    data = base64.b64encode(r.read()).decode()
                videos.append({"filename": vid["filename"], "data": data, "node_id": node_id})

        return {"images": images, "videos": videos}

    @modal.method()
    def health(self):
        return {"status": "ok"}

    @modal.method()
    def list_models(self):
        import os

        vol.reload()

        # Standalone folders shown as their own sections
        solo_folders = ["loras", "vae", "controlnet", "upscale_models",
                        "embeddings", "clip", "text_encoders"]
        result = {}
        for folder in solo_folders:
            folder_path = os.path.join(MODELS_PATH, folder)
            if not os.path.isdir(folder_path):
                result[folder] = []
                continue
            files = []
            for fname in sorted(os.listdir(folder_path)):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    files.append({"name": fname, "size": os.path.getsize(fpath), "folder": folder})
            result[folder] = files
        # Checkpoint-family folders all shown under "checkpoints" in the sidebar,
        # but each file carries its real "folder" so inject/delete uses the right path.
        checkpoint_family = ["checkpoints", "diffusion_models", "unet"]
        result["checkpoints"] = []
        for folder in checkpoint_family:
            folder_path = os.path.join(MODELS_PATH, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    result["checkpoints"].append({"name": fname, "size": os.path.getsize(fpath), "folder": folder})
        return result

    @modal.method()
    def delete_model(self, folder: str, filename: str):
        import os
        safe_folder = os.path.basename(folder)
        safe_file = os.path.basename(filename)
        target = os.path.join(MODELS_PATH, safe_folder, safe_file)
        if not os.path.isfile(target):
            return {"status": "error", "message": "File not found"}
        os.remove(target)
        vol.commit()
        return {"status": "ok", "deleted": f"{safe_folder}/{safe_file}"}


# NOTE: max_inputs=1 on purpose. Video generations nearly fill GPU VRAM,
# so sharing one snapshotted container between concurrent inputs risks OOM.
# Worse, cancelling one input shuts down the whole container and kills the
# other in-flight inputs (Modal reschedules them, causing error cascades).
# Extra prompts scale out to separate containers instead.
@app.cls(
    gpu="a10g",
    cpu=4,
    memory=16384,
    timeout=3600,
    min_containers=0,
    # Keep containers warm for 60s to avoid cold-start costs during iterative workflows
    scaledown_window=60,
    volumes={MODELS_PATH: vol, CUSTOM_NODES_PATH: custom_nodes_vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class ComfyAPI(_ComfyAPIMixin):
    pass


@app.cls(
    gpu="a100",
    cpu=4,
    memory=32768,
    timeout=3600,
    min_containers=0,
    # Keep containers warm for 60s to avoid cold-start costs during iterative workflows
    scaledown_window=60,
    volumes={MODELS_PATH: vol, CUSTOM_NODES_PATH: custom_nodes_vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class ComfyAPI_A100(_ComfyAPIMixin):
    pass


@app.cls(
    gpu="t4",
    cpu=2,
    memory=8192,
    timeout=3600,
    min_containers=0,
    # Keep containers warm for 60s to avoid cold-start costs during iterative workflows
    scaledown_window=60,
    volumes={MODELS_PATH: vol, CUSTOM_NODES_PATH: custom_nodes_vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class ComfyAPI_T4(_ComfyAPIMixin):
    pass
