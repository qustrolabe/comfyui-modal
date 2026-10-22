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
# 2.0.14: live event relay. The container can now report ComfyUI's live
# events (progress / executing / preview frames) back to the browser while the
# prompt is running, and it returns its own status messages so the queue panel
# can show the real execution duration. See run_prompt_stream().
COMFYAPP_VERSION = "2.0.14"

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

# ComfyUI's binary websocket event types (comfy_api/protocol.py).
BINARY_EVENT_PREVIEW_IMAGE = 1
BINARY_EVENT_UNENCODED_PREVIEW_IMAGE = 2
BINARY_EVENT_TEXT = 3
BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA = 4

# The text events that are relayed to the browser while the prompt runs.
#
# Deliberately NOT relayed: execution_start / execution_success /
# execution_error / execution_interrupted / executed / execution_cached.
# The comfyui-modal node already emits those itself (it owns the local
# prompt lifecycle), so relaying the container's copies would make the
# browser see every one of them twice.
#
# "progress_state" is the important one: it is the per-node progress map
# (comfy_execution/progress.py:161-186) that drives the progress bar on the
# canvas in the 0.42.1 frontend. "progress" is the older flat
# {value, max, node} variant and is relayed as well so both consumers are fed.
# "executing" carries the node that is actually running, plus display_node
# (execution.py:497) - the frontend reads `data.display_node || data.node`.
RELAYED_TEXT_EVENTS = ("progress", "progress_state", "executing")

# Feature flags the container-side websocket client announces. Without
# "supports_preview_metadata" ComfyUI downgrades the preview frames to
# UNENCODED_PREVIEW_IMAGE, which carries no node metadata - the browser then
# cannot attach the preview to the right node
# (comfy_execution/progress.py:207-229).
CONTAINER_CLIENT_FEATURE_FLAGS = {
    "supports_preview_metadata": True,
    "supports_progress_text_metadata": True,
}

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

        # NOTE: a volume reload fails with "there are open files preventing the
        # operation" when ComfyUI holds a model file open (mmap'ed safetensors).
        # That must NOT abort the enter handler - an exception there kills the
        # container and turns one failed start into an endless container churn.
        # The mount still holds the current contents, so just log and continue.
        try:
            vol.reload()
        except Exception as e:
            print(f"[comfyui-modal] NOTE: models volume reload skipped ({e})")
        try:
            custom_nodes_vol.reload()
        except Exception as e:
            print(f"[comfyui-modal] NOTE: custom-nodes volume reload skipped ({e})")
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

    def _prompt_extra_data(self, preview_method: str = "") -> dict:
        """The extra_data that is sent to the container's own ComfyUI.

        `preview_method` is the only thing we forward, and it is what makes
        live previews possible at all: ComfyUI resets its global preview
        method to the CLI default (--preview-method none) for EVERY prompt
        unless the prompt carries an override
        (execution.py:734 -> latent_preview.py:142-148). Without it the
        container renders no preview frames whatsoever.

        Nothing else from the browser's extra_data is forwarded: it also
        carries the user's ComfyUI.org credentials (auth_token_comfy_org /
        api_key_comfy_org) and the whole workflow under extra_pnginfo - none
        of that may be shipped to a third-party cloud.
        """
        extra = {}
        if preview_method and preview_method != "default":
            extra["preview_method"] = preview_method
        return extra

    def _submit_prompt_to_comfy(self, workflow, input_images, prompt_id, client_id, extra_data):
        """POST a prompt to the container's own ComfyUI and return its reply.

        `prompt_id` is the id the LOCAL ComfyUI already handed to the
        browser. ComfyUI 0.42.1 accepts a client-supplied prompt id
        (server.py:1179-1193, validated by comfy_execution/jobs.py:34-51) as
        long as it is a UUID in canonical lowercase-hyphenated form - which is
        exactly what str(uuid.uuid4()) produces. Re-using it means every event
        the container emits already carries the id the browser filters on, so
        nothing has to be rewritten on the way back.
        """
        import urllib.request
        import urllib.error
        from pathlib import Path

        if input_images:
            stage_remote_input_images(
                Path("/root/comfy/ComfyUI/input"),
                input_images,
            )

        body = {"prompt": workflow, "client_id": client_id}
        if prompt_id:
            body["prompt_id"] = prompt_id
        if extra_data:
            body["extra_data"] = extra_data
        payload = json.dumps(body).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{COMFYUI_API_PORT}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
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

    @modal.method()
    def run_prompt(self, workflow: dict, input_images: dict = None,
                   prompt_id: str = "", client_id: str = "",
                   preview_method: str = "") -> dict:
        """Run a prompt and return the finished outputs (no live events).

        This is the ORIGINAL, non-streaming entry point and it is kept on
        purpose:
          * an older comfyui-modal node (one that was loaded before this
            change) keeps working against a newly deployed app, and
          * the node has a fallback for a deployed app that does not have
            run_prompt_stream yet (i.e. one that was not redeployed after
            this change).
        The node prefers run_prompt_stream(), which additionally relays
        ComfyUI's live events while the prompt runs.
        """
        if not client_id:
            client_id = str(uuid.uuid4())
        queued = self._submit_prompt_to_comfy(
            workflow,
            input_images,
            prompt_id,
            client_id,
            self._prompt_extra_data(preview_method),
        )
        return self._poll_until_done(queued["prompt_id"], client_id)

    def _poll_until_done(self, prompt_id: str, client_id: str) -> dict:
        """Wait for the prompt to finish and collect its outputs.

        The returned dict also carries the container's own `status.messages`.
        Those are the timestamps ComfyUI stamped on the real execution
        (execution.py:680-685) and they are the most accurate duration source
        available: they bracket the actual GPU work and exclude Modal's queue
        and RPC overhead. The shape is exactly what the frontend needs - a list
        of [event_name, data] pairs where data["timestamp"] is in milliseconds
        (comfy_execution/jobs.py:226-238).
        """
        import urllib.request
        delay = 0.5
        elapsed = 0.0
        while elapsed < 3600:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{COMFYUI_API_PORT}/history/{prompt_id}"
            ) as r:
                history = json.loads(r.read())
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status") or {}
                messages = status.get("messages", []) or []
                if status.get("status_str") == "error":
                    # The prompt was accepted but failed while executing.
                    # Raising here (instead of returning empty outputs) is
                    # what lets the browser show the error and lets the queue
                    # panel record an "execution_error" timestamp.
                    raise RuntimeError(self._describe_error(messages))
                outputs = entry.get("outputs", {})
                return {**self._collect_outputs(outputs), "messages": messages}
            time.sleep(delay)
            elapsed += delay
            delay = min(delay * 1.5, 5.0)
        raise TimeoutError(f"Prompt {prompt_id} timed out")

    @staticmethod
    def _describe_error(messages) -> str:
        """Human-readable text for a failed prompt, taken from its messages."""
        for entry in messages or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] == "execution_error":
                data = entry[1]
                if isinstance(data, dict):
                    return str(
                        data.get("exception_message")
                        or data.get("message")
                        or "the remote prompt failed"
                    )
        return "the remote prompt failed"

    # ------------------------------------------------------------------
    # Live event relay
    # ------------------------------------------------------------------

    @modal.method()
    async def run_prompt_stream(self, workflow: dict, input_images: dict = None,
                                prompt_id: str = "", client_id: str = "",
                                preview_method: str = "") -> None:
        """Run a prompt and stream ComfyUI's live events back to the caller.

        Yields a sequence of dicts:
          {"kind": "event",   "event": <name>, "data": <payload>}
          {"kind": "preview", "metadata": {...}, "image": <encoded bytes>}
          {"kind": "result",  "result": {"images": [...], "videos": [...],
                                         "messages": [...]}}

        WHY A GENERATOR: a Modal generator method streams its yielded values
        back over the SAME RPC that runs the function, so the events come from
        the very container that is executing the prompt. A separate polling
        method would have to land on the same container - the app runs with
        @modal.concurrent(max_inputs=1) - which Modal does not guarantee.
        The generator removes that problem by construction.

        The container-side websocket is opened BEFORE the prompt is
        submitted, so not a single event is missed.
        """
        import asyncio
        import aiohttp

        if not client_id:
            client_id = str(uuid.uuid4())
        if not prompt_id:
            prompt_id = str(uuid.uuid4())

        queue: asyncio.Queue = asyncio.Queue()

        async with aiohttp.ClientSession() as session:
            url = f"http://127.0.0.1:{COMFYUI_API_PORT}/ws?clientId={client_id}"
            async with session.ws_connect(url) as ws:
                # The feature-flag handshake MUST be the first message on the
                # socket: ComfyUI only records the flags from the first
                # message (server.py:359-381). Without it the preview frames
                # lose their node metadata and the browser cannot attach them
                # to the right node.
                await ws.send_str(json.dumps({
                    "type": "feature_flags",
                    "data": CONTAINER_CLIENT_FEATURE_FLAGS,
                }))

                reader = asyncio.create_task(self._ws_event_reader(ws, queue))
                try:
                    # The submission is a blocking urllib call, so it runs in
                    # a thread - otherwise it would stall the event loop and
                    # with it the websocket reader.
                    queued = await asyncio.to_thread(
                        self._submit_prompt_to_comfy,
                        workflow,
                        input_images,
                        prompt_id,
                        client_id,
                        self._prompt_extra_data(preview_method),
                    )
                except Exception:
                    reader.cancel()
                    raise

                remote_prompt_id = queued.get("prompt_id", prompt_id)

                async def watch_completion():
                    try:
                        result = await asyncio.to_thread(
                            self._poll_until_done, remote_prompt_id, client_id
                        )
                        await queue.put({"kind": "result", "result": result})
                    except Exception as e:
                        await queue.put({"kind": "error", "error": str(e) or repr(e)})

                watcher = asyncio.create_task(watch_completion())

                try:
                    while True:
                        item = await queue.get()
                        kind = item.get("kind")
                        if kind == "error":
                            raise RuntimeError(item.get("error", "the remote prompt failed"))
                        if kind == "result":
                            # The last couple of frames ("executing: null",
                            # "execution_success") are sent right before the
                            # history entry is written, so give the reader a
                            # moment to pick them up before finishing.
                            await asyncio.sleep(0.3)
                            tail = []
                            while True:
                                try:
                                    extra = queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                                if extra.get("kind") not in ("result", "error"):
                                    tail.append(extra)
                            for extra in tail:
                                yield extra
                            yield item
                            break
                        yield item
                except asyncio.CancelledError:
                    # The browser cancelled the prompt. Tell the container's
                    # ComfyUI to stop, so it does not keep sampling (and
                    # burning the GPU) until the prompt finishes on its own.
                    # A plain call, not an await: the task is being
                    # cancelled, so anything awaited here could be cancelled
                    # again before it runs. It is a local HTTP call and takes
                    # a couple of milliseconds.
                    self._interrupt_comfy(remote_prompt_id)
                    raise
                finally:
                    reader.cancel()
                    watcher.cancel()

    def _interrupt_comfy(self, prompt_id: str = "") -> None:
        """Ask the container's own ComfyUI to stop the running prompt.

        Without this, cancelling a prompt in the browser only cancels the
        LOCAL side: the container's ComfyUI keeps sampling until the prompt
        finishes on its own, burning the GPU (Modal's own cancel() does not
        stop the process either - it only releases the input slot).
        """
        import urllib.request
        import urllib.error

        body = {"prompt_id": prompt_id} if prompt_id else {}
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{COMFYUI_API_PORT}/interrupt",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            print(f"[comfyui-modal] interrupted the remote prompt {prompt_id}")
        except Exception as e:
            print(f"[comfyui-modal] could not interrupt the remote prompt: {e}")

    async def _ws_event_reader(self, ws, queue):
        """Read the container's own ComfyUI websocket and queue its events.

        Text frames are the JSON events ("progress", "progress_state",
        "executing", ...). Binary frames are the preview images; ComfyUI
        frames them as [4-byte event type][payload]
        (server.py:1398-1404, :1431-1466).
        """
        import asyncio
        import aiohttp

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    if len(data) < 4:
                        continue
                    event = int.from_bytes(data[:4], "big")
                    payload = data[4:]
                    if event == BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA:
                        item = self._decode_preview_frame(payload)
                        if item is not None:
                            await queue.put(item)
                    # The other binary frames (plain preview images) are NOT
                    # relayed: without the node metadata the browser cannot
                    # attach them to a node, so they would only be noise.
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        obj = json.loads(msg.data)
                    except Exception:
                        continue
                    event = obj.get("type")
                    if event in RELAYED_TEXT_EVENTS:
                        await queue.put({
                            "kind": "event",
                            "event": event,
                            "data": obj.get("data"),
                        })
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[comfyui-modal] event reader stopped: {e}")

    @staticmethod
    def _decode_preview_frame(payload: bytes):
        """Split a type-4 preview frame into its metadata and image bytes.

        Wire format (server.py:1431-1466):
            [4-byte metadata length][metadata JSON][image bytes]
        The image is already JPEG-encoded by the container's ComfyUI, so it is
        shipped as-is and decoded on the other side.
        """
        if len(payload) < 4:
            return None
        meta_len = int.from_bytes(payload[:4], "big")
        try:
            metadata = json.loads(payload[4:4 + meta_len].decode("utf-8"))
        except Exception:
            return None
        image = payload[4 + meta_len:]
        if not image:
            return None
        return {"kind": "preview", "metadata": metadata, "image": image}

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

        # NOTE: Modal refuses to refresh a volume while any file inside it is
        # open by a process in the container ("there are open files preventing
        # the operation"). ComfyUI keeps model files open (mmap'ed safetensors)
        # for as long as the model is loaded, so this reload fails whenever a
        # generation has a model resident. The listing is still correct - the
        # files are right there on the mount - so treat a failed reload as
        # "keep the current view" instead of failing the whole request. It
        # used to raise, the route turned that into a 503, and the sidebar
        # then claimed the app was not deployed yet.
        try:
            vol.reload()
        except Exception as e:
            print(f"[comfyui-modal] NOTE: volume reload skipped ({e}); listing the current view")

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
        # NOTE: the commit hits the same "open files preventing the operation"
        # conflict as the volume reload when ComfyUI still holds another model
        # file open. Report it clearly instead of raising a bare Modal error.
        try:
            vol.commit()
        except Exception as e:
            return {
                "status": "error",
                "message": (
                    f"Removed from the container view, but the volume commit failed: {e}. "
                    "ComfyUI is holding a model file open - unload/close it and retry."
                ),
            }
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
