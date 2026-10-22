import asyncio
import functools
import modal

_apis = {
    "a10g": modal.Cls.from_name("comfyui", "ComfyAPI"),
    "a100": modal.Cls.from_name("comfyui", "ComfyAPI_A100"),
    "t4":   modal.Cls.from_name("comfyui", "ComfyAPI_T4"),
}
_download_fn = modal.Function.from_name("comfyui", "download_model_to_volume")
_batch_download_fn = modal.Function.from_name("comfyui", "batch_download_models")
_sync_custom_nodes_fn = modal.Function.from_name("comfyui", "sync_custom_nodes_to_volume")
_get_volume_status_fn = modal.Function.from_name("comfyui", "get_volume_status")
_upload_model_fn = modal.Function.from_name("comfyui", "upload_model_to_volume")
_upload_model_chunk_fn = modal.Function.from_name("comfyui", "upload_model_chunk")

_current_gpu = "a10g"
_api_instances = {}


def set_gpu(gpu: str):
    global _current_gpu
    if gpu in _apis:
        _current_gpu = gpu


def get_gpu() -> str:
    return _current_gpu


def _api():
    if _current_gpu not in _api_instances:
        _api_instances[_current_gpu] = _apis[_current_gpu]()
    return _api_instances[_current_gpu]


def clear_cache():
    """Clear cached API instance handles so subsequent requests use fresh handles."""
    _api_instances.clear()


def _modal_error_handler(func):
    """Decorator that catches common exceptions and re-raises with user-friendly messages."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (ConnectionError, OSError) as e:
            raise ConnectionError(
                "Modal connection failed. Check your internet connection and Modal token."
            ) from e
        except TimeoutError as e:
            raise TimeoutError(
                "Modal request timed out. The container may be cold-starting (1-3 min)."
            ) from e
        except Exception as e:
            if getattr(type(e), "__module__", "").startswith("modal"):
                raise RuntimeError(
                    f"Modal error: {e}. Try redeploying with the Deploy button."
                ) from e
            raise
    return wrapper


def _is_missing_method(exc: Exception) -> bool:
    """True when the error means "the deployed app cannot stream a prompt".

    Kept for reference: the fallback now triggers on ANY failure that happens
    before the container acknowledged the prompt, because that is the only
    case where running the prompt a second time is safe.
    """
    if type(exc).__name__ in ("NotFoundError", "InvalidError"):
        return True
    text = str(exc).lower()
    if "has no method" in text or "non-generator" in text:
        return True
    if "generator" in text and (
        "not supported" in text or "unsupported" in text
        or "not implemented" in text or "unknown" in text
    ):
        return True
    return False


async def _run_streaming(api, workflow, input_images, prompt_id, client_id, preview_method, on_event):
    """Consume the container's streaming prompt method.

    Returns the result dict, or None when the streaming path could not be
    used - the caller then falls back to the plain (non-streaming) method.
    """
    try:
        stream_fn = api.run_prompt_stream
    except Exception as e:
        # The deployed app predates run_prompt_stream.
        print(f"[comfyui-modal] no streaming prompt method: {type(e).__name__}: {e}")
        return None

    got_item = False
    result = None
    stream = stream_fn.remote_gen(workflow, input_images, prompt_id, client_id, preview_method)
    try:
        async for item in stream:
            kind = item.get("kind")
            if kind == "result":
                # Do not break out of the loop: let the remote stream end on
                # its own so the RPC is closed cleanly.
                result = item.get("result")
                continue
            if kind == "started":
                # The container has the prompt now - from here on a failure
                # must NOT be retried with the plain method, because that
                # would run the prompt twice.
                got_item = True
                continue
            got_item = True
            if on_event is not None:
                await on_event(item)
    except Exception as e:
        if got_item:
            # The prompt is already running on the container; retrying it
            # would generate it twice. Let the error through.
            raise
        # Nothing came back at all, so the prompt was (almost certainly)
        # never submitted - fall back to the plain method instead of failing
        # the run.
        import traceback
        print(f"[comfyui-modal] the streaming prompt method failed, using the "
              f"plain one (live progress and previews stay off): "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    finally:
        try:
            await stream.aclose()
        except Exception:
            pass

    if result is None:
        raise RuntimeError("Modal returned no result for the prompt")
    return result


@_modal_error_handler
async def run_prompt(workflow: dict, input_images: dict = None, prompt_id: str = "",
                     client_id: str = "", preview_method: str = "",
                     on_event=None) -> dict:
    """Run a prompt on Modal.

    `on_event` is an optional async callback that receives the container's live
    events as they arrive:
        {"kind": "event",   "event": <name>, "data": <payload>}
        {"kind": "preview", "metadata": {...}, "image": <bytes>}
    """
    api = _api()
    result = await _run_streaming(
        api, workflow, input_images or {}, prompt_id, client_id, preview_method, on_event
    )
    if result is not None:
        return result

    # Fallback: the deployed app does not have the streaming method (it was
    # not redeployed after the feature was added) or the deployed Modal version
    # cannot run generators. The prompt still runs and its images still come
    # back - only the live progress and the previews are missing.
    #
    # The legacy call keeps the ORIGINAL two-argument signature on purpose:
    # that is the signature the currently deployed app has, and the new one
    # accepts it as well.
    return await asyncio.to_thread(
        lambda: api.run_prompt.remote(workflow, input_images or {}),
    )


@_modal_error_handler
async def get_object_info() -> dict:
    return await asyncio.to_thread(lambda: _api().object_info.remote())


@_modal_error_handler
async def health_check() -> dict:
    return await asyncio.to_thread(lambda: _api().health.remote())


@_modal_error_handler
async def download_model(url: str, filename: str, save_path: str = "checkpoints", hf_token: str = "") -> dict:
    return await asyncio.to_thread(
        lambda: _download_fn.remote(url=url, filename=filename, save_path=save_path, hf_token=hf_token),
    )


@_modal_error_handler
async def batch_download_models(items: list, hf_token: str = "") -> list:
    return await asyncio.to_thread(
        lambda: _batch_download_fn.remote(items, hf_token=hf_token),
    )


@_modal_error_handler
async def list_models() -> dict:
    return await asyncio.to_thread(lambda: _api().list_models.remote())


@_modal_error_handler
async def delete_model(folder: str, filename: str) -> dict:
    return await asyncio.to_thread(
        lambda: _api().delete_model.remote(folder=folder, filename=filename),
    )


@_modal_error_handler
async def sync_custom_nodes(archive_data: bytes) -> dict:
    size_mb = len(archive_data) / 1024**2
    print(f"[comfyui-modal] Calling Modal sync_custom_nodes_to_volume ({size_mb:.1f} MB)... "
          f"first call may build the image (several minutes).")
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: _sync_custom_nodes_fn.remote(archive_data),
            ),
            timeout=600,
        )
    except TimeoutError:
        raise TimeoutError(
            "Sync timed out after 10 min. The Modal image may still be building — "
            "wait a few minutes and retry. Check `modal app logs comfyui` for progress."
        ) from None
    print(f"[comfyui-modal] Modal sync returned: {result.get('status') if isinstance(result, dict) else result}")
    return result


@_modal_error_handler
async def get_sync_status() -> dict:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                lambda: _get_volume_status_fn.remote(),
            ),
            timeout=180,
        )
    except TimeoutError:
        raise TimeoutError(
            "Volume status timed out after 3 min. The Modal image may still be building "
            "(first call takes several minutes) — wait and retry."
        ) from None


@_modal_error_handler
async def upload_model_to_volume(file_data: bytes, folder: str, filename: str) -> dict:
    return await asyncio.to_thread(
        lambda: _upload_model_fn.remote(file_data, folder, filename),
    )


@_modal_error_handler
async def upload_model_chunk(chunk_data: bytes, folder: str, filename: str, offset: int, is_last: bool) -> dict:
    return await asyncio.to_thread(
        lambda: _upload_model_chunk_fn.remote(chunk_data, folder, filename, offset, is_last),
    )
