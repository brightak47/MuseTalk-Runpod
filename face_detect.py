"""A commercially-licensed stand-in for the face detector LivePortrait expects.

LivePortrait's code and weights are MIT, but its `Cropper` builds an `insightface` FaceAnalysis to find the
face and produce initial landmarks. InsightFace's *pretrained* models (buffalo_l: det_10g.onnx and
2d106det.onnx) are released for non-commercial research only, and a third party redistributing them under
their own MIT card does not change that. So they are not downloaded and not used here.

What InsightFace actually contributes is small. Reading `cropper.crop_source_image`, the detector's entire
output is one `Nx2` landmark array, used twice:

    crop_image(img, lmk, ...)                  -> computes the crop box, scale and roll
    human_landmark_runner.run(img_rgb, lmk)    -> LivePortrait's own MIT landmark.onnx refines it

The precise landmarks come from LivePortrait itself. The detector only has to say roughly where the face is.

YuNet (OpenCV Zoo, MIT, ~340 KB) does that and nothing else. It is used in preference to MediaPipe, which
was tried first: `import mediapipe` pulls in TensorFlow, which then failed against the numpy this image
needs -- MuseTalk, LivePortrait and TensorFlow each want a different numpy, and that triangle has no
solution worth maintaining. YuNet runs through `cv2.FaceDetectorYN`, which OpenCV already provides here.

The fit is exact rather than approximate. YuNet returns five landmarks ordered
`[right eye, left eye, nose tip, right mouth corner, left mouth corner]`, and LivePortrait's
`parse_pt2_from_pt_x` dispatches on point count to `parse_pt2_from_pt5`, which reads `[0]` and `[1]` as the
eyes and `[3]` and `[4]` as the mouth corners. So the five points go straight through and every line of
LivePortrait's own geometry runs unmodified -- no forked crop maths, no reindexing, nothing to drift.
"""

import os
from pathlib import Path

import numpy as np

# Downloaded by download_weights.py. Kept beside the other weights rather than in a cache directory so the
# build-time check can see it.
YUNET_MODEL = Path(os.environ.get("YUNET_MODEL", "/app/MuseTalk/models/yunet/face_detection_yunet.onnx"))


class _Face:
    """The one attribute `cropper` reads off a detected face."""

    def __init__(self, landmarks: np.ndarray, bbox: np.ndarray, score: float):
        # Named for the interface it replaces. Five points, not 106 -- which LivePortrait handles natively.
        self.landmark_2d_106 = landmarks
        self.bbox = bbox
        self.det_score = score


class YuNetFaceAnalysis:
    """
    Drop-in for `FaceAnalysisDIY`: the same four methods the Cropper calls, none of the licence.

    Constructed with InsightFace's keyword arguments (`name`, `root`, `providers`) so it can be swapped in
    without touching LivePortrait. They are accepted and ignored.
    """

    def __init__(self, name: str = "buffalo_l", root: str = "", providers=None, **kwargs):
        self._detector = None
        self._score_threshold = 0.6
        self._size = (320, 320)

    def prepare(self, ctx_id: int = 0, det_size=(512, 512), det_thresh: float = 0.1, **kwargs):
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                f"this OpenCV ({cv2.__version__}) has no FaceDetectorYN, so YuNet cannot run. "
                "FaceDetectorYN arrived in OpenCV 4.5.4."
            )
        if not YUNET_MODEL.is_file():
            raise RuntimeError(f"YuNet model missing at {YUNET_MODEL}; download_weights.py should fetch it")

        # det_thresh comes from LivePortrait's config and is low (0.1-0.15) because InsightFace scores
        # differently. YuNet is confident on a portrait, so the floor is raised to avoid locking onto
        # texture in a background; a portrait with no face should be an error, not a guess.
        self._score_threshold = max(0.5, float(det_thresh))
        self._size = (int(det_size[0]), int(det_size[1]))
        self._detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", self._size, self._score_threshold, 0.3, 5000
        )
        return self

    def warmup(self):
        if self._detector is None:
            self.prepare()
        # Grey rather than black: a pure-black frame can fail detection outright, and a warmup exists to
        # pay the first-call cost, not to assert anything about the result.
        self.get(np.full((320, 320, 3), 128, dtype=np.uint8))

    def get(self, img_bgr: np.ndarray, **kwargs):
        """Return faces ordered as LivePortrait expects, or an empty list when there is no face."""
        if self._detector is None:
            self.prepare()

        height, width = img_bgr.shape[:2]
        # The detector must be told the frame size, and portraits here vary.
        self._detector.setInputSize((width, height))
        _retval, detections = self._detector.detect(img_bgr)
        if detections is None or len(detections) == 0:
            return []

        faces = []
        for row in detections:
            x, y, w, h = (float(v) for v in row[:4])
            # Columns 4..13 are the five landmarks as x,y pairs, already in pixels and already in the
            # order parse_pt2_from_pt5 expects.
            five = np.array(row[4:14], dtype=np.float32).reshape(5, 2)
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            faces.append(_Face(five, bbox, float(row[14])))

        direction = kwargs.get("direction", "large-small")
        max_face_num = int(kwargs.get("max_face_num", 0) or 0)
        faces = _sorted(faces, direction)
        return faces[:max_face_num] if max_face_num > 0 else faces


def _sorted(faces, direction: str):
    """Mirror LivePortrait's own ordering so the same face is chosen as before."""
    def area(f):
        x0, y0, x1, y1 = f.bbox
        return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))

    if direction == "small-large":
        return sorted(faces, key=area)
    if direction == "left-right":
        return sorted(faces, key=lambda f: f.bbox[0])
    if direction == "right-left":
        return sorted(faces, key=lambda f: -f.bbox[0])
    if direction == "up-down":
        return sorted(faces, key=lambda f: f.bbox[1])
    if direction == "down-up":
        return sorted(faces, key=lambda f: -f.bbox[1])
    return sorted(faces, key=area, reverse=True)  # "large-small", LivePortrait's default


def install(liveportrait_root: str) -> None:
    """
    Make LivePortrait's Cropper use this instead of InsightFace.

    Patched on the module LivePortrait imports the name into, rather than by editing the checkout, so an
    upgrade cannot quietly resurrect the InsightFace path. If the attribute ever stops existing this raises
    here, at startup, instead of failing inside a paid job -- or worse, succeeding by loading weights we are
    not licensed to use.
    """
    import sys

    if liveportrait_root not in sys.path:
        sys.path.insert(0, liveportrait_root)

    from src.utils import cropper as lp_cropper

    if not hasattr(lp_cropper, "FaceAnalysisDIY"):
        raise RuntimeError(
            "LivePortrait's cropper no longer exposes FaceAnalysisDIY, so the InsightFace replacement "
            "cannot be installed. Check src/utils/cropper.py before running: proceeding would either use "
            "non-commercial InsightFace weights or fail inside a job."
        )
    lp_cropper.FaceAnalysisDIY = YuNetFaceAnalysis
