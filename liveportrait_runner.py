"""LivePortrait: turn one still portrait into a short clip of a presenter who looks alive.

This is the first half of a presentation video. MuseTalk syncs a mouth to audio and touches nothing else, so
driving it with a still gives a presenter who never blinks for the length of a lesson. LivePortrait supplies
that, and MuseTalk then syncs the mouth onto the result.

Blinking only, by default. Head sway was built first and tried at four amplitudes; none of them looked right.
A sine has none of the reasons a real head moves -- emphasis, thought, addressing someone -- so small
amplitudes were invisible and large ones read as a wobble. A blink has no such problem, because a blink
really is periodic and involuntary. The sway knob remains for anyone who disagrees, defaulted to zero.

The economics are why it is here rather than a video model. This is a 512x512 warping network, so the idle
clip costs cents and is generated once per avatar and cached; Wan 2.2 cost $0.086-$0.129 for the same job.
The per-minute cost of a finished presentation stays MuseTalk's measured $0.08.

Two choices worth knowing about:

`animation_region` defaults to "pose". LivePortrait can transfer expression, lips, eyes or everything, and
transferring the driving clip's mouth would fight MuseTalk for control of the same pixels. Pose alone is
head movement only, which composes cleanly. "all" adds blinks and expression at the cost of that conflict,
and `flag_normalize_lip` closes the mouth first to reduce it; both are exposed rather than chosen here.

InsightFace is not used. face_detect.install() replaces the detector with YuNet (OpenCV Zoo, MIT) before
the pipeline is built -- see that module for why the substitution is small, and why YuNet rather than
MediaPipe, which could not be reconciled with this image's numpy.
"""

import os
from pathlib import Path

LIVEPORTRAIT_DIR = Path(os.environ.get("LIVEPORTRAIT_DIR", "/app/LivePortrait"))
# Where download_weights.py puts LivePortrait's MIT weights, which is not where LivePortrait looks by default.
WEIGHTS = Path(os.environ.get("LIVEPORTRAIT_WEIGHTS", "/app/MuseTalk/models/liveportrait"))

_pipeline = None

# Every InferenceConfig field animate() sets. Named here so the build gate can check them against the real
# dataclass without duplicating the list, since a rename shows up as a default silently taking over.
INFERENCE_FIELDS = (
    "flag_eye_retargeting",
    "flag_lip_retargeting",
    "flag_relative_motion",
    "animation_region",
    "flag_normalize_lip",
    "flag_stitching",
    "flag_pasteback",
    "flag_do_crop",
    "driving_option",
    "driving_multiplier",
)


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

    import face_detect

    # Before anything imports the Cropper, so the InsightFace class is never the one constructed.
    face_detect.install(str(LIVEPORTRAIT_DIR))
    if str(LIVEPORTRAIT_DIR) not in sys.path:
        sys.path.insert(0, str(LIVEPORTRAIT_DIR))

    from src.config.crop_config import CropConfig
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.utils.cropper import Cropper

    # Keep landmark.onnx on the CPU.
    #
    # LivePortraitPipeline builds `Cropper(crop_cfg=crop_cfg)` without forwarding a device flag, so the
    # Cropper defaults to CUDA and wants onnxruntime-gpu. That package is pinned to 1.18 upstream, which
    # expects CUDA 12, while this image is CUDA 11.7 -- matching wheels to toolkits here is a whole class of
    # breakage for no benefit. landmark.onnx runs once per source portrait (the driving video is not
    # cropped), so CPU costs a few hundred milliseconds per avatar, paid once and cached.
    #
    # Patched as a subclass on the module the pipeline resolves the name from, the same way the detector is,
    # so neither the checkout nor the upstream call site is edited.
    import src.live_portrait_pipeline as lp_pipeline

    class _CpuCropper(Cropper):
        def __init__(self, **kwargs):
            kwargs.setdefault("flag_force_cpu", True)
            super().__init__(**kwargs)

    lp_pipeline.Cropper = _CpuCropper

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

    Blinks require `eye_retargeting`, and that forces absolute motion. The reason is a single line in
    LivePortrait's pipeline: once any retargeting is on, the relative-motion path rebuilds the keypoints as

        x_d_i_new = x_s + eyes_delta + lip_delta

    which discards the pose term computed just above it, so the head stops moving. The absolute path is

        x_d_i_new = x_d_i_new + eyes_delta + lip_delta

    and keeps both. Absolute motion is safe here only because idle_template() is built from this portrait's
    own latents, so R_new = R_d_i and t_new are already the source's own orientation plus the sway -- for a
    template taken from somebody else's face it would transfer their head position outright.

    Upstream marks flag_eye_retargeting "not recommend to be True, WIP". It is used anyway because it is the
    only path that reads c_eyes_lst, but that is a reason to look at the output rather than trust it.
    """
    import sys

    if str(LIVEPORTRAIT_DIR) not in sys.path:
        sys.path.insert(0, str(LIVEPORTRAIT_DIR))
    from src.config.argument_config import ArgumentConfig

    out_dir.mkdir(parents=True, exist_ok=True)

    eye_retargeting = bool(options.get("eye_retargeting", True))
    # Forced rather than merely defaulted: the two together are the trap described above, and the failure is
    # silent -- a clip that looks fine until you notice the head never moves.
    relative_motion = False if eye_retargeting else bool(options.get("relative_motion", True))

    pipe = pipeline()

    # Where the behaviour flags actually have to go.
    #
    # execute() reads args for exactly three things -- source, driving and flag_crop_driving_video -- and
    # takes every other decision from inf_cfg, which is the InferenceConfig the pipeline was constructed
    # with. Upstream's inference.py bridges the two with
    #
    #     inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    #
    # before constructing the pipeline. Building the pipeline directly, as this module does, skips that
    # step, so flags passed only in ArgumentConfig are accepted, ignored, and every run silently uses the
    # defaults. That is why an earlier build could set flag_eye_retargeting and still never blink, and why
    # animation_region appeared to do nothing. The pipeline is cached per worker and jobs are serial, so
    # applying them per call is safe.
    cfg = pipe.live_portrait_wrapper.inference_cfg
    wanted = {
        # The only place c_eyes_lst is ever read.
        "flag_eye_retargeting": eye_retargeting,
        # Left off: MuseTalk owns the mouth, and turning this on would hand it to LivePortrait.
        "flag_lip_retargeting": False,
        "flag_relative_motion": relative_motion,
        # Pose only by default: see the module docstring on why the mouth is left for MuseTalk. With a
        # template driver this cannot transfer expression in any case -- LivePortrait gates that branch on
        # flag_is_driving_video -- but it still selects which of scale, t and R are taken from the template.
        "animation_region": str(options.get("animation_region", "pose")),
        "flag_normalize_lip": bool(options.get("normalize_lip", False)),
        "flag_stitching": bool(options.get("stitching", True)),
        "flag_pasteback": bool(options.get("pasteback", True)),
        "flag_do_crop": bool(options.get("do_crop", True)),
        "driving_option": str(options.get("driving_option", "expression-friendly")),
        "driving_multiplier": float(options.get("driving_multiplier", 1.0)),
    }
    assert set(wanted) == set(INFERENCE_FIELDS), "INFERENCE_FIELDS is out of step with animate()"
    for field, value in wanted.items():
        # Checked rather than assumed: a renamed field would otherwise set a harmless new attribute and
        # leave the real one at its default, which is precisely the failure this block exists to fix.
        if not hasattr(cfg, field):
            raise RuntimeError(
                f"InferenceConfig has no {field!r}; LivePortrait would silently run with its default. "
                "Check src/config/inference_config.py before running."
            )
        setattr(cfg, field, value)

    args = ArgumentConfig(
        source=str(source_image),
        driving=str(driving),
        output_dir=str(out_dir),
        # The one behaviour flag execute() really does read off args.
        flag_crop_driving_video=bool(options.get("crop_driving_video", False)),
        det_thresh=float(options.get("det_thresh", 0.15)),
        scale=float(options.get("scale", 2.3)),
        vy_ratio=float(options.get("vy_ratio", -0.125)),
    )

    animated, _side_by_side = pipe.execute(args)
    produced = Path(animated)
    if not produced.is_file():
        raise RuntimeError(f"LivePortrait reported {produced} but it does not exist")
    return produced


def idle_template(source_image: Path, seconds: float, fps: int, out_path: Path, **options) -> Path:
    """
    Make a driving template of a presenter sitting still: blinks, no head movement, mouth untouched.

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
    from src.utils.retargeting_utils import calc_eye_close_ratio

    pipe = pipeline()
    wrapper = pipe.live_portrait_wrapper

    img_rgb = load_image_rgb(str(source_image))
    crop = pipe.cropper.crop_source_image(img_rgb, pipe.cropper.crop_cfg)
    if crop is None:
        raise RuntimeError("no face found in the source portrait")

    # The eye-open value LivePortrait compares against is an aspect ratio from the 203-point landmarks, not
    # an abstract 0-1 openness. Measuring this portrait's own ratio makes the retarget a true no-op between
    # blinks and self-calibrates per face, where a hardcoded constant would nudge every avatar's eyes
    # towards whatever number happened to be chosen.
    open_ratio = float(calc_eye_close_ratio(crop["lmk_crop"][None])[0][0])

    prepared = wrapper.prepare_source(crop["img_crop_256x256"])
    info = wrapper.get_kp_info(prepared)
    x_s = wrapper.transform_keypoint(info)

    n_frames = max(1, int(round(seconds * fps)))
    # Off by default. A sinusoidal sway is the obvious way to synthesise head motion and it does not survive
    # being watched: a real presenter's head moves because they are emphasising something, and a smooth cycle
    # that never does reads as a slow wobble instead. Blinking has no such problem -- it genuinely is periodic
    # and involuntary -- so that is what is left. The knob stays because the machinery is the same either way.
    sway_deg = float(options.get("sway_degrees", 0.0))
    sway_period = float(options.get("sway_seconds", 4.0))
    blink_every = float(options.get("blink_seconds", 4.0))  # roughly a natural resting blink rate
    blink_frames = max(2, int(round(0.12 * fps)))           # ~120 ms, about how long a blink takes
    # Not 0.0: a fully-collapsed ratio asks the retargeting network for a state it never saw in training.
    closed_ratio = float(options.get("blink_closed_ratio", 0.02))

    # With no sway the rotation is the source's own, identical in every frame, so it is built once rather than
    # recomputed per frame from a sine that is always zero -- and, more to the point, every frame then gets a
    # bit-identical R instead of one that differs in the last decimal place and jitters.
    still_head = sway_deg == 0.0

    base_pitch = float(info["pitch"].detach().cpu().numpy().reshape(-1)[0])
    base_yaw = float(info["yaw"].detach().cpu().numpy().reshape(-1)[0])
    base_roll = float(info["roll"].detach().cpu().numpy().reshape(-1)[0])

    def as_numpy(t):
        return t.detach().cpu().numpy().astype(np.float32)

    import torch

    def rotation(pitch, yaw, roll):
        return get_rotation_matrix(
            torch.tensor([[pitch]], dtype=torch.float32),
            torch.tensor([[yaw]], dtype=torch.float32),
            torch.tensor([[roll]], dtype=torch.float32),
        )

    still_R = rotation(base_pitch, base_yaw, base_roll).cpu().numpy().astype(np.float32) if still_head else None

    template = {"n_frames": n_frames, "output_fps": fps, "motion": [], "c_eyes_lst": [], "c_lip_lst": []}
    for i in range(n_frames):
        if still_head:
            R_np = still_R
        else:
            phase = 2.0 * np.pi * i / max(1.0, sway_period * fps)
            # Yaw and pitch on different periods so the motion never repeats on an obvious beat.
            yaw = base_yaw + sway_deg * np.sin(phase)
            pitch = base_pitch + (sway_deg * 0.4) * np.sin(phase * 0.7 + 1.1)
            roll = base_roll + (sway_deg * 0.2) * np.sin(phase * 0.5)
            R_np = rotation(pitch, yaw, roll).cpu().numpy().astype(np.float32)

        template["motion"].append({
            "scale": as_numpy(info["scale"]),
            "R": R_np,
            "exp": as_numpy(info["exp"]),
            "t": as_numpy(info["t"]),
            "kp": as_numpy(info["kp"]),
            "x_s": as_numpy(x_s),
        })

        # Eyes open normally; closed for a few frames at each blink.
        #
        # Shaped (1, 2) to match what LivePortrait's own calc_ratio() produces, because calc_combined_eye_ratio
        # indexes c_d_eyes_i[0][0]. A flat two-element array raises IndexError there, which is why the earlier
        # version of this file could never have blinked even with retargeting switched on.
        into_blink = i % max(1, int(round(blink_every * fps)))
        closed = into_blink < blink_frames
        ratio = closed_ratio if closed else open_ratio
        template["c_eyes_lst"].append(np.array([[ratio, ratio]], dtype=np.float32))
        # The mouth stays shut: MuseTalk owns it from here. Shaped (1, 1) to match calc_ratio() as well,
        # though nothing reads it while flag_lip_retargeting is off.
        template["c_lip_lst"].append(np.array([[0.0]], dtype=np.float32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        pickle.dump(template, handle)
    return out_path
