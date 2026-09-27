"""RunPod serverless worker for MuseTalk 1.5 lip-sync.

Input (job["input"]):
    audio_url | audio_b64      required   narration to lip-sync to
    video_url | image_url | video_b64 | image_b64
                               required   the face to drive (a looping idle clip, or a still)
    fps                        optional   default 25
    bbox_shift                 optional   default 0 (v1 only; v15 uses its own offset)
    extra_margin               optional   default 10
    parsing_mode               optional   "jaw" (default) or "raw"
    output_key                 optional   destination key, e.g. "lessons/42/slice_000.mp4"
    project                    optional   echoed back for cost attribution

Output:
    {"video_url": ...}   when R2/S3 is configured (R2_BUCKET + credentials in env)
    {"video_path": ...}  when a network volume is mounted and no bucket is set
    {"video_b64": ...}   otherwise, for results under INLINE_LIMIT_MB
    plus {"seconds": ..., "project": ...}

Environment (set on the endpoint, not baked into the image):
    R2_ACCOUNT_ID / R2_BUCKET / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY   -> upload to R2
    S3_ENDPOINT_URL / S3_BUCKET / AWS_* also work for any S3-compatible store
"""

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import requests
import runpod

MUSETALK_DIR = Path("/app/MuseTalk")
INLINE_LIMIT_MB = int(os.environ.get("INLINE_LIMIT_MB", "18"))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "120"))
VOLUME_DIR = Path("/runpod-volume")


class InputError(Exception):
    """Something wrong with the request rather than the worker."""


def _fetch(url: str, dest: Path) -> Path:
    """Download a URL to dest. Times out rather than parking a paid GPU forever."""
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    if dest.stat().st_size == 0:
        raise InputError(f"downloaded file is empty: {url}")
    return dest


def _decode(data: str, dest: Path) -> Path:
    dest.write_bytes(base64.b64decode(data))
    return dest


def _resolve_media(job_input: dict, kind: str, work: Path) -> Path:
    """kind is 'audio', 'video' or 'image'; accepts <kind>_url or <kind>_b64."""
    suffix = {"audio": ".wav", "video": ".mp4", "image": ".png"}[kind]
    dest = work / f"input_{kind}{suffix}"
    if job_input.get(f"{kind}_url"):
        return _fetch(job_input[f"{kind}_url"], dest)
    if job_input.get(f"{kind}_b64"):
        return _decode(job_input[f"{kind}_b64"], dest)
    return None


def _upload(path: Path, key: str):
    """Upload to R2/S3 when configured; returns a URL or None."""
    bucket = os.environ.get("R2_BUCKET") or os.environ.get("S3_BUCKET")
    if not bucket:
        return None

    import boto3

    account = os.environ.get("R2_ACCOUNT_ID")
    endpoint = os.environ.get("S3_ENDPOINT_URL") or (
        f"https://{account}.r2.cloudflarestorage.com" if account else None
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
    )
    content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})

    public_base = os.environ.get("R2_PUBLIC_BASE_URL")
    if public_base:
        return f"{public_base.rstrip('/')}/{key}"
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=7 * 24 * 3600
    )


def handler(job):
    job_input = job.get("input") or {}
    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="musetalk_", dir="/tmp"))

    try:
        audio = _resolve_media(job_input, "audio", work)
        if audio is None:
            raise InputError("audio_url or audio_b64 is required")

        face = _resolve_media(job_input, "video", work) or _resolve_media(job_input, "image", work)
        if face is None:
            raise InputError("one of video_url, video_b64, image_url or image_b64 is required")

        # MuseTalk reads its tasks from a YAML config rather than CLI arguments
        result_dir = work / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        config_path = work / "task.yaml"
        config_path.write_text(
            "task_0:\n"
            f'  video_path: "{face}"\n'
            f'  audio_path: "{audio}"\n',
            encoding="utf-8",
        )

        command = [
            "python", "-m", "scripts.inference",
            "--inference_config", str(config_path),
            "--result_dir", str(result_dir),
            "--version", "v15",
            "--fps", str(int(job_input.get("fps", 25))),
            "--extra_margin", str(int(job_input.get("extra_margin", 10))),
            "--parsing_mode", str(job_input.get("parsing_mode", "jaw")),
            "--bbox_shift", str(int(job_input.get("bbox_shift", 0))),
            "--ffmpeg_path", os.environ.get("FFMPEG_PATH", "/usr/bin"),
        ]
        if job_input.get("use_float16", True):
            command.append("--use_float16")

        completed = subprocess.run(
            command, cwd=str(MUSETALK_DIR), capture_output=True, text=True, timeout=int(job_input.get("timeout", 3600))
        )
        if completed.returncode != 0:
            # Surface the real reason — the tail of stderr is what actually explains failures.
            return {
                "error": "MuseTalk inference failed",
                "returncode": completed.returncode,
                "stderr": completed.stderr[-4000:],
                "stdout": completed.stdout[-1000:],
            }

        produced = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not produced:
            return {
                "error": "inference finished but produced no video",
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
        output = produced[-1]

        response = {
            "seconds": round(time.time() - started, 1),
            "size_bytes": output.stat().st_size,
        }
        if job_input.get("project"):
            response["project"] = job_input["project"]

        key = job_input.get("output_key") or f"musetalk/{uuid.uuid4()}.mp4"
        url = _upload(output, key)
        if url:
            response["video_url"] = url
            response["output_key"] = key
        elif VOLUME_DIR.is_dir():
            destination = VOLUME_DIR / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, destination)
            response["video_path"] = str(destination)
        elif output.stat().st_size <= INLINE_LIMIT_MB * 1024 * 1024:
            response["video_b64"] = base64.b64encode(output.read_bytes()).decode()
        else:
            return {
                "error": (
                    f"result is {output.stat().st_size // (1024 * 1024)} MB with no R2 bucket or "
                    "network volume configured; set R2_BUCKET or attach a volume"
                )
            }
        return response

    except InputError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"could not fetch an input: {exc}"}
    except subprocess.TimeoutExpired:
        return {"error": "inference exceeded its timeout"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
