"""LivePortrait: turn one still portrait into a short clip of a presenter who looks alive.

This is the first half of a presentation video. MuseTalk syncs a mouth to audio but cannot invent head
motion, so driving it with a still gives a presenter who never moves. LivePortrait supplies that motion --
head sway, and blinks if asked for -- and MuseTalk then syncs the mouth onto the result.

The economics are why it is here rather than a video model. This is a 512x512 warping network, so the idle
clip costs cents and is generated once per avatar and cached; Wan 2.2 cost $0.086-$0.129 for the same job.
The per-minute cost of a finished presentation stays MuseTalk's measured $0.08.

Two choices worth knowing about:

`animation_region` defaults to "pose". LivePortrait can transfer expression, lips, eyes or everything, and
transferring the driving clip's mouth would fight MuseTalk for control of the same pixels. Pose alone is
head movement only, which composes cleanly. "all" adds blinks and expression at the cost of that conflict,
and `flag_normalize_lip` closes the mouth first to reduce it; both are exposed rather than chosen here.

InsightFace is not used. mediapipe_face.install() replaces the detector before the pipeline is built -- see
that module for why the substitution is small and what it would mean to get it wrong.
"""

import os
from pathlib import Path

LIVEPORTRAIT_DIR = Path(os.environ.get("LIVEPORTRAIT_DIR", "/app/LivePortrait"))
# Where download_weights.py puts LivePortrait's MIT weights, which is not where LivePortrait looks by default.
WEIGHTS = Path(os.environ.get("LIVEPORTRAIT_WEIGHTS", "/app/MuseTalk/models/liveportrait"))

_pipeline = None


def pipeline():
    """
    Build the pipeline once per worker.

    Lazily, because a job that only lip-syncs should not pay for loading an animation model, and because
    importing LivePortrait pulls in torch and onnxruntime.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import sys

    import mediapipe_face

    # Before anything imports the Cropper, so the InsightFace class is never the one constructed.
    mediapipe_face.install(str(LIVEPORTRAIT_DIR))
    if str(LIVEPORTRAIT_DIR) not in sys.path:
        sys.path.insert(0, str(LIVEPORTRAIT_DIR))

    from src.config.crop_config import CropConfig
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

    missing = [
        name for name in (
            "base_models/appearance_feature_extractor.pth",
            "base_models/motion_extractor.pth",
            "base_models/spade_generator.pth",
            "base_models/warping_module.pth",
            "retargeting_models/stitching_retargeting_module.pth",
            "landmark.onnx",
        )
        if not (WEIGHTS / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"LivePortrait weights missing under {WEIGHTS}: {', '.join(missing)}")

    inference_cfg = InferenceConfig(
        checkpoint_F=str(WEIGHTS / "base_models/appearance_feature_extractor.pth"),
        checkpoint_M=str(WEIGHTS / "base_models/motion_extractor.pth"),
        checkpoint_G=str(WEIGHTS / "base_models/spade_generator.pth"),
        checkpoint_W=str(WEIGHTS / "base_models/warping_module.pth"),
        checkpoint_S=str(WEIGHTS / "retargeting_models/stitching_retargeting_module.pth"),
    )
    crop_cfg = CropConfig(landmark_ckpt_path=str(WEIGHTS / "landmark.onnx"))
    # insightface_root is left at its default on purpose: nothing reads it now, and pointing it at a
    # directory that does not exist is a clearer signal than pointing it somewhere plausible.

    _pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    return _pipeline


def animate(source_image: Path, driving: Path, out_dir: Path, **options) -> Path:
    """
    Animate `source_image` with the motion in `driving`, which may be a video or a .pkl motion template.

    A template is the better input for an avatar: it is numeric pose data rather than footage, so it carries
    no likeness, is deterministic between runs, and costs nothing to store. `execute` returns the animation
    and a side-by-side comparison; only the animation is wanted.
    """
    import sys

    if str(LIVEPORTRAIT_DIR) not in sys.path:
        sys.path.insert(0, str(LIVEPORTRAIT_DIR))
    from src.config.argument_config import ArgumentConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    args = ArgumentConfig(
        source=str(source_image),
        driving=str(driving),
        output_dir=str(out_dir),
        # Pose only by default: see the module docstring on why the mouth is left for MuseTalk.
        animation_region=str(options.get("animation_region", "pose")),
        flag_normalize_lip=bool(options.get("normalize_lip", False)),
        # Relative motion means a template made from any face produces the same movement on this portrait,
        # which is what makes one template reusable across an avatar library.
        flag_relative_motion=bool(options.get("relative_motion", True)),
        driving_option=str(options.get("driving_option", "expression-friendly")),
        driving_multiplier=float(options.get("driving_multiplier", 1.0)),
        flag_stitching=bool(options.get("stitching", True)),
        flag_pasteback=bool(options.get("pasteback", True)),
        flag_do_crop=bool(options.get("do_crop", True)),
        flag_crop_driving_video=bool(options.get("crop_driving_video", False)),
        det_thresh=float(options.get("det_thresh", 0.15)),
        scale=float(options.get("scale", 2.3)),
        vy_ratio=float(options.get("vy_ratio", -0.125)),
        flag_use_half_precision=bool(options.get("half_precision", True)),
    )

    animated, _side_by_side = pipeline().execute(args)
    produced = Path(animated)
    if not produced.is_file():
        raise RuntimeError(f"LivePortrait reported {produced} but it does not exist")
    return produced


def idle_template(source_image: Path, seconds: float, fps: int, out_path: Path, **options) -> Path:
    """
    Make a driving template of a presenter sitting still: slow head sway, optional blinks, mouth untouched.

    Built from the source portrait's own latents rather than from footage, so the face never drifts towards
    somebody else's: every frame reuses the source's scale, expression, keypoints and translation, and only
    the rotation matrix varies. Blinks ride on `c_eyes_lst`, which is the channel LivePortrait's own eye
    retargeting reads.

    The result is numeric, deterministic, loopable, and carries no likeness from any driving clip.
    """
    import pickle
    import sys

    import numpy as np

    if str(LIVEPORTRAIT_DIR) not in sys.path:
        sys.path.insert(0, str(LIVEPORTRAIT_DIR))
    from src.utils.camera import get_rotation_matrix
    from src.utils.io import load_image_rgb

    pipe = pipeline()
    wrapper = pipe.live_portrait_wrapper

    img_rgb = load_image_rgb(str(source_image))
    crop = pipe.cropper.crop_source_image(img_rgb, pipe.cropper.crop_cfg)
    if crop is None:
        raise RuntimeError("no face found in the source portrait")

    prepared = wrapper.prepare_source(crop["img_crop_256x256"])
    info = wrapper.get_kp_info(prepared)
    x_s = wrapper.transform_keypoint(info)

    n_frames = max(1, int(round(seconds * fps)))
    sway_deg = float(options.get("sway_degrees", 2.0))      # a presenter's head, not a metronome
    sway_period = float(options.get("sway_seconds", 6.0))   # one slow cycle; faster reads as fidgeting
    blink_every = float(options.get("blink_seconds", 4.0))  # roughly a natural resting blink rate
    blink_frames = max(2, int(round(0.12 * fps)))           # ~120 ms, about how long a blink takes

    base_pitch = float(info["pitch"].detach().cpu().numpy().reshape(-1)[0])
    base_yaw = float(info["yaw"].detach().cpu().numpy().reshape(-1)[0])
    base_roll = float(info["roll"].detach().cpu().numpy().reshape(-1)[0])

    def as_numpy(t):
        return t.detach().cpu().numpy().astype(np.float32)

    import torch

    template = {"n_frames": n_frames, "output_fps": fps, "motion": [], "c_eyes_lst": [], "c_lip_lst": []}
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / max(1.0, sway_period * fps)
        # Yaw and pitch on different periods so the motion never repeats on an obvious beat.
        yaw = base_yaw + sway_deg * np.sin(phase)
        pitch = base_pitch + (sway_deg * 0.4) * np.sin(phase * 0.7 + 1.1)
        roll = base_roll + (sway_deg * 0.2) * np.sin(phase * 0.5)

        R = get_rotation_matrix(
            torch.tensor([[pitch]], dtype=torch.float32),
            torch.tensor([[yaw]], dtype=torch.float32),
            torch.tensor([[roll]], dtype=torch.float32),
        )

        template["motion"].append({
            "scale": as_numpy(info["scale"]),
            "R": R.cpu().numpy().astype(np.float32),
            "exp": as_numpy(info["exp"]),
            "t": as_numpy(info["t"]),
            "kp": as_numpy(info["kp"]),
            "x_s": as_numpy(x_s),
        })

        # Eyes open normally; closed for a few frames at each blink. Two values, one per eye.
        into_blink = i % max(1, int(round(blink_every * fps)))
        closed = into_blink < blink_frames
        openness = 0.0 if closed else 0.38
        template["c_eyes_lst"].append(np.array([openness, openness], dtype=np.float32))
        # The mouth stays shut: MuseTalk owns it from here.
        template["c_lip_lst"].append(np.array([0.0], dtype=np.float32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        pickle.dump(template, handle)
    return out_path
