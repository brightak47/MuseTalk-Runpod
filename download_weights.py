"""Fetch MuseTalk's inference weights at build time.

Uses huggingface_hub's Python API rather than the CLI: `huggingface-cli` was renamed to `hf` in
recent releases, so shelling out to it breaks the build (exit 127) depending on the version pulled.

Deliberately omitted, and both omissions are licence decisions rather than size ones:

  * MuseTalk's syncnet checkpoint (ByteDance/LatentSync, OpenRAIL++) is only used for training.
  * InsightFace's buffalo_l models (det_10g.onnx, 2d106det.onnx), which LivePortrait's Cropper would
    otherwise load. InsightFace releases its pretrained models for non-commercial research only, and the
    fact that KwaiVGI's weights repo redistributes them under its own MIT card does not change their terms.
    face_detect.py replaces them with YuNet (OpenCV Zoo, MIT); see it for why that substitution is small.

LivePortrait's own weights ARE included and are MIT: the four base models, the stitching/retargeting
module, and landmark.onnx, which is what actually produces the precise landmarks.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = Path("models")

# (repo_id, file in repo, destination under models/)
DOWNLOADS = [
    ("TMElyralab/MuseTalk", "musetalk/musetalk.json", "musetalk/musetalk.json"),
    ("TMElyralab/MuseTalk", "musetalk/pytorch_model.bin", "musetalk/pytorch_model.bin"),
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "musetalkV15/musetalk.json"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "musetalkV15/unet.pth"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae/config.json"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "sd-vae/diffusion_pytorch_model.bin"),
    ("openai/whisper-tiny", "config.json", "whisper/config.json"),
    ("openai/whisper-tiny", "pytorch_model.bin", "whisper/pytorch_model.bin"),
    ("openai/whisper-tiny", "preprocessor_config.json", "whisper/preprocessor_config.json"),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "dwpose/dw-ll_ucoco_384.pth"),
    # LivePortrait (MIT) -- animates a still portrait so the presenter has head motion and blinks before
    # MuseTalk syncs the mouth. Note what is NOT here: insightface/models/buffalo_l/*.
    ("KwaiVGI/LivePortrait", "liveportrait/base_models/appearance_feature_extractor.pth", "liveportrait/base_models/appearance_feature_extractor.pth"),
    ("KwaiVGI/LivePortrait", "liveportrait/base_models/motion_extractor.pth", "liveportrait/base_models/motion_extractor.pth"),
    ("KwaiVGI/LivePortrait", "liveportrait/base_models/spade_generator.pth", "liveportrait/base_models/spade_generator.pth"),
    ("KwaiVGI/LivePortrait", "liveportrait/base_models/warping_module.pth", "liveportrait/base_models/warping_module.pth"),
    ("KwaiVGI/LivePortrait", "liveportrait/retargeting_models/stitching_retargeting_module.pth", "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
    ("KwaiVGI/LivePortrait", "liveportrait/landmark.onnx", "liveportrait/landmark.onnx"),
    # Mirror of the BiSeNet face-parsing checkpoint that upstream fetches from Google Drive.
    # Same file, WTFPL-licensed, and no gdown/quota fragility in the build.
    ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", "face-parse-bisent/79999_iter.pth"),
]

RESNET18_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
RESNET18_DEST = "face-parse-bisent/resnet18-5c106cde.pth"


def fetch(repo_id: str, filename: str, destination: str) -> Path:
    target = MODELS / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copyfile(cached, target)
    print(f"  {repo_id}/{filename} -> {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    return target


# YuNet, the face detector that stands in for InsightFace. MIT, about 340 KB, and not on HuggingFace, so
# it is fetched by URL the same way the face-parse weight is.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def fetch_yunet() -> Path:
    destination = MODELS / "yunet" / "face_detection_yunet.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 100_000:
        print(f"  yunet already present ({destination.stat().st_size / 1e3:.0f} KB)")
        return destination
    with urllib.request.urlopen(YUNET_URL, timeout=120) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)
    size = destination.stat().st_size
    if size < 100_000:
        sys.exit(f"YuNet model looks wrong: {size} bytes from {YUNET_URL}")
    print(f"  yunet -> {destination} ({size / 1e3:.0f} KB)")
    return destination


def main() -> int:
    print("downloading MuseTalk inference weights", flush=True)
    for repo_id, filename, destination in DOWNLOADS:
        fetch(repo_id, filename, destination)

    target = MODELS / RESNET18_DEST
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(RESNET18_URL, target)
    print(f"  {RESNET18_URL} -> {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)

    yunet = fetch_yunet()

    # Fail the build here rather than at the first inference request.
    missing = [d for _, _, d in DOWNLOADS if not (MODELS / d).is_file()]
    if not target.is_file():
        missing.append(RESNET18_DEST)
    if not yunet.is_file():
        missing.append(str(yunet))
    if missing:
        print("missing after download: " + ", ".join(missing), file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p in MODELS.rglob("*") if p.is_file())
    print(f"weights ready: {total / 1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
