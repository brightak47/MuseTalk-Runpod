"""Build-time proof that LivePortrait is usable and that InsightFace is absent.

Both halves matter. The first is the ordinary check that an import works. The second is a licence
guarantee: InsightFace's pretrained models are non-commercial, LivePortrait's Cropper loads them by
default, and a future rebuild that quietly pulled them in would be an infringement rather than a bug. So
the build fails if those files appear on disk, and fails if the YuNet replacement cannot be installed.
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

# 3. YuNet, and the substitution actually taking hold.
sys.path.insert(0, "/app/MuseTalk")
import cv2  # noqa: E402

if not hasattr(cv2, "FaceDetectorYN"):
    sys.exit(f"OpenCV {cv2.__version__} has no FaceDetectorYN; YuNet needs 4.5.4 or newer")
print(f"opencv {cv2.__version__} with FaceDetectorYN")

import face_detect  # noqa: E402

if not face_detect.YUNET_MODEL.is_file():
    sys.exit(f"YuNet model missing at {face_detect.YUNET_MODEL}")
print(f"yunet model present ({face_detect.YUNET_MODEL.stat().st_size / 1e3:.0f} KB)")

# Construct it for real: a model file that exists but will not load should fail the build.
face_detect.YuNetFaceAnalysis().prepare(det_size=(320, 320), det_thresh=0.6)
print("YuNet detector loads")

face_detect.install(str(LP))
sys.path.insert(0, str(LP))
from src.utils import cropper as lp_cropper  # noqa: E402

if lp_cropper.FaceAnalysisDIY is not face_detect.YuNetFaceAnalysis:
    sys.exit("the YuNet replacement did not take hold; the Cropper would still load InsightFace")
print("Cropper detector is YuNetFaceAnalysis")

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

# 6. The two upstream behaviours the blink support rests on. Both failed silently once already: eye
# retargeting read a channel we were filling with the wrong shape, so the blinks simply never appeared and
# nothing complained. These assert the contract instead of trusting it.
import inspect  # noqa: E402

cfg_fields = ArgumentConfig.__dataclass_fields__
for field in ("flag_eye_retargeting", "flag_lip_retargeting", "flag_relative_motion", "animation_region"):
    if field not in cfg_fields:
        sys.exit(f"ArgumentConfig no longer has {field}; animate() passes it and would raise")

import src.live_portrait_pipeline as lp_pipeline  # noqa: E402
import src.live_portrait_wrapper as lp_wrapper  # noqa: E402

pipeline_src = inspect.getsource(lp_pipeline.LivePortraitPipeline.execute)
# The absolute-motion branch is what keeps head pose alive while retargeting is on. If upstream collapses
# these two branches, blinks would start costing us the sway, and only a human watching would notice.
if "x_d_i_new = x_d_i_new + \\" not in pipeline_src:
    sys.exit("the absolute-motion retargeting branch is gone; enabling eye retargeting would kill head pose")
if "flag_eye_retargeting and source_lmk is not None" not in pipeline_src:
    sys.exit("c_eyes_lst is no longer gated the way idle_template assumes; blinks may silently stop")
print("retargeting branch preserves head pose")

ratio_src = inspect.getsource(lp_wrapper.LivePortraitWrapper.calc_combined_eye_ratio)
# idle_template writes (1, 2) arrays precisely because this indexes twice. A flat array raises IndexError.
if "c_d_eyes_i[0][0]" not in ratio_src:
    sys.exit("calc_combined_eye_ratio changed its indexing; the c_eyes_lst shape in idle_template is now wrong")
print("c_eyes_lst shape contract holds")

# 7. The flags animate() sets must exist on InferenceConfig, because that -- not ArgumentConfig -- is where
# execute() reads them. Setting an unknown name would quietly leave LivePortrait on its default, which is
# how an entire build shipped with eye retargeting "enabled" and no blink in any frame.
import liveportrait_runner  # noqa: E402

missing_cfg = [f for f in liveportrait_runner.INFERENCE_FIELDS if not hasattr(InferenceConfig, f)]
if missing_cfg:
    sys.exit(f"InferenceConfig is missing {missing_cfg}; animate() would run with defaults instead")
execute_src = inspect.getsource(lp_pipeline.LivePortraitPipeline.execute)
if "inf_cfg.flag_eye_retargeting" not in execute_src:
    sys.exit("execute() no longer reads flag_eye_retargeting off inf_cfg; recheck where flags belong")
print(f"all {len(liveportrait_runner.INFERENCE_FIELDS)} inference flags exist where execute() reads them")

print("liveportrait verification passed")
