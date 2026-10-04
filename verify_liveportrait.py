"""Build-time proof that LivePortrait is usable and that InsightFace is absent.

Both halves matter. The first is the ordinary check that an import works. The second is a licence
guarantee: InsightFace's pretrained models are non-commercial, LivePortrait's Cropper loads them by
default, and a future rebuild that quietly pulled them in would be an infringement rather than a bug. So
the build fails if those files appear on disk, and fails if the MediaPipe replacement cannot be installed.
"""

import sys
from pathlib import Path

LP = Path("/app/LivePortrait")
WEIGHTS = Path("/app/MuseTalk/models/liveportrait")

# 1. The weights we are entitled to use.
required = [
    "base_models/appearance_feature_extractor.pth",
    "base_models/motion_extractor.pth",
    "base_models/spade_generator.pth",
    "base_models/warping_module.pth",
    "retargeting_models/stitching_retargeting_module.pth",
    "landmark.onnx",
]
missing = [name for name in required if not (WEIGHTS / name).is_file()]
if missing:
    sys.exit(f"LivePortrait weights missing under {WEIGHTS}: {', '.join(missing)}")
print(f"LivePortrait weights present: {len(required)} files")

# 2. The weights we are not. Anything matching these is a licence problem, wherever it came from.
strays = []
for root in (Path("/app"), Path("/root/.insightface"), WEIGHTS.parent):
    if not root.exists():
        continue
    for pattern in ("**/det_10g.onnx", "**/2d106det.onnx", "**/buffalo_l/**"):
        strays.extend(str(p) for p in root.glob(pattern) if p.is_file())
if strays:
    sys.exit(
        "InsightFace pretrained models found, which are licensed for non-commercial research only: "
        + ", ".join(sorted(set(strays))[:5])
    )
print("no InsightFace models on disk")

# 3. MediaPipe, and the substitution actually taking hold.
sys.path.insert(0, "/app/MuseTalk")
import mediapipe  # noqa: E402

print("mediapipe", mediapipe.__version__)

import mediapipe_face  # noqa: E402

mediapipe_face.install(str(LP))
sys.path.insert(0, str(LP))
from src.utils import cropper as lp_cropper  # noqa: E402

if lp_cropper.FaceAnalysisDIY is not mediapipe_face.MediaPipeFaceAnalysis:
    sys.exit("the MediaPipe replacement did not take hold; the Cropper would still load InsightFace")
print("Cropper detector is MediaPipeFaceAnalysis")

# 4. LivePortrait's own geometry accepts five points, which is what makes the substitution legitimate.
import numpy as np  # noqa: E402
from src.utils.crop import parse_pt2_from_pt_x, parse_rect_from_landmark  # noqa: E402

five = np.array([[205, 215], [305, 215], [256, 256], [225, 307], [287, 307]], dtype=np.float32)
pt2 = parse_pt2_from_pt_x(five, use_lip=True)
assert pt2.shape == (2, 2), pt2.shape
assert pt2[1][1] > pt2[0][1], "lip centre should be below eye centre"
parse_rect_from_landmark(five, scale=2.3, vy_ratio=-0.125)
print("five-point landmarks drive LivePortrait's crop geometry")

# 5. The pipeline modules import. Model loading needs a GPU, so it is not attempted here.
from src.config.argument_config import ArgumentConfig  # noqa: E402,F401
from src.config.crop_config import CropConfig  # noqa: E402,F401
from src.config.inference_config import InferenceConfig  # noqa: E402,F401
from src.live_portrait_pipeline import LivePortraitPipeline  # noqa: E402,F401

print("LivePortrait pipeline imports")
print("liveportrait verification passed")
