"""RunPod serverless worker for MuseTalk 1.5 lip-sync.

Input (job["input"]):
    audio_url | audio_b64      required   narration to lip-sync to
    video_url | image_url | video_b64 | image_b64
                               required   the face to drive (a looping idle clip, or a still)
    fps                        optional   default 25
    bbox_shift                 optional   default 0 (v1 only; v15 uses its own offset)
    extra_margin               optional   default 10
    parsing_mode               optional   "jaw" (default) or "raw"

    mode                       optional   "lipsync" (default) | "animate" | "present"

      lipsync   what this worker always did: sync a mouth onto a face you supply.
      animate   LivePortrait only: one still portrait -> a short clip with head motion. No audio needed.
                Done once per avatar and cached; the result is what `lipsync` should be driven with.
      present   both, in one job: portrait + audio -> animated idle -> lip-synced presentation video.
                Chaining them here rather than across two endpoints saves a cold start and an upload of
                the intermediate clip, which for a 15 GB image is most of the wall clock.

    seconds                    animate/present: length of the idle clip, default 6
    fps                        default 25
    animation_region           "pose" (default), "all", "exp", "eyes", "lip" -- see liveportrait_runner
    driving_url | driving_b64  optional: your own driving video or .pkl template instead of generated idle
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
    suffix = {"audio": ".wav", "video": ".mp4", "image": ".png", "driving": ".pkl"}[kind]
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


def _animate(job_input: dict, portrait: Path, work: Path) -> Path:
    """
    One still portrait -> a short clip of a presenter who looks alive.

    The motion comes from a generated template rather than footage unless a driving file is supplied: it is
    numeric pose data, so it carries no likeness, is identical between runs, and is built from this
    portrait's own latents so the face cannot drift towards someone else's.
    """
    import liveportrait_runner as lp

    fps = int(job_input.get("fps", 25))
    seconds = float(job_input.get("seconds", 6))

    driving = _resolve_media(job_input, "driving", work)
    if driving is None:
        driving = lp.idle_template(
            portrait, seconds=seconds, fps=fps, out_path=work / "idle.pkl",
            sway_degrees=float(job_input.get("sway_degrees", 2.0)),
            sway_seconds=float(job_input.get("sway_seconds", 6.0)),
            blink_seconds=float(job_input.get("blink_seconds", 4.0)),
        )

    return lp.animate(
        portrait, driving, work / "animated",
        animation_region=job_input.get("animation_region", "pose"),
        normalize_lip=job_input.get("normalize_lip", False),
        driving_multiplier=job_input.get("driving_multiplier", 1.0),
    )


def _deliver(output: Path, job_input: dict, started: float, work: Path, extra: dict | None = None) -> dict:
    """
    Hand a finished video back the same way whichever mode produced it.

    R2 when it is configured, the network volume when one is attached, and base64 only for something small
    -- a presentation is minutes long and will not fit in a response, so a bucket is effectively required
    for `present`.
    """
    response = {
        "seconds": round(time.time() - started, 1),
        "size_bytes": output.stat().st_size,
    }
    response.update(extra or {})
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
                f"result is {output.stat().st_size // (1024 * 1024)} MB with no R2 bucket or network "
                "volume configured; set R2_BUCKET or attach a volume"
            )
        }
    return response


def handler(job):
    job_input = job.get("input") or {}
    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="musetalk_", dir="/tmp"))

    try:
        mode = str(job_input.get("mode", "lipsync"))
        if mode not in {"lipsync", "animate", "present"}:
            raise InputError(f"unknown mode {mode!r}; use lipsync, animate or present")

        face = _resolve_media(job_input, "video", work) or _resolve_media(job_input, "image", work)
        if face is None:
            raise InputError("one of video_url, video_b64, image_url or image_b64 is required")

        # animate and present both begin by giving a still portrait some life. present then hands the
        # result to MuseTalk as the face to lip-sync, which is the whole point of chaining them here.
        animated = None
        if mode in {"animate", "present"}:
            animated = _animate(job_input, face, work)
            if mode == "animate":
                return _deliver(animated, job_input, started, work, extra={"mode": "animate"})
            face = animated

        audio = _resolve_media(job_input, "audio", work)
        if audio is None:
            raise InputError("audio_url or audio_b64 is required")

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

        # inference.py's argparse default for --unet_config points at models/musetalk/config.json,
        # a filename upstream's own download script never produces. Pass the real paths.
        version = str(job_input.get("version", "v15"))
        weights = "musetalkV15" if version == "v15" else "musetalk"
        unet_name = "unet.pth" if version == "v15" else "pytorch_model.bin"

        command = [
            "python", "-m", "scripts.inference",
            "--inference_config", str(config_path),
            "--result_dir", str(result_dir),
            "--version", version,
            "--unet_config", f"./models/{weights}/musetalk.json",
            "--unet_model_path", f"./models/{weights}/{unet_name}",
            "--whisper_dir", "./models/whisper",
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

        return _deliver(output, job_input, started, work, extra={"mode": mode})

    except InputError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"could not fetch an input: {exc}"}
    except subprocess.TimeoutExpired:
        return {"error": "inference exceeded its timeout"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
