"""A commercially-licensed stand-in for the face detector LivePortrait expects.

LivePortrait's code and weights are MIT, but its `Cropper` builds an `insightface` FaceAnalysis to find the
face and produce initial landmarks. InsightFace's *pretrained* models (buffalo_l: det_10g.onnx and
2d106det.onnx) are released for non-commercial research only, and a third party redistributing them under
their own MIT card does not change that. So they are not downloaded and not used here.

What InsightFace actually contributes is small. Reading `cropper.crop_source_image`, the detector's entire
output is one `Nx2` landmark array, which is used twice:

    crop_image(img, lmk, ...)                  -> computes the crop box, scale and roll
    human_landmark_runner.run(img_rgb, lmk)    -> LivePortrait's own MIT landmark.onnx refines it

The precise landmarks come from LivePortrait itself. The detector only has to say roughly where the face is.

And LivePortrait already supports a five-point layout: `parse_pt2_from_pt_x` dispatches on the number of
points, and `parse_pt2_from_pt5` reads `[0]` and `[1]` as the eyes and `[3]` and `[4]` as the mouth corners.
MediaPipe Face Mesh (Apache 2.0) gives those directly. So this returns five points and every line of
LivePortrait's own geometry runs unmodified -- no forked crop maths, no reindexing, nothing to drift.
"""

import numpy as np

# Face Mesh landmark indices. Eyes are averaged across their corners rather than taken from a single point,
# because a corner moves with a blink and the crop should not.
LEFT_EYE = (33, 133)
RIGHT_EYE = (362, 263)
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


class _Face:
    """The one attribute `cropper` reads off a detected face."""

    def __init__(self, landmarks: np.ndarray, bbox: np.ndarray, score: float):
        # Named for the interface it replaces. Five points, not 106 -- which LivePortrait handles natively.
        self.landmark_2d_106 = landmarks
        self.bbox = bbox
        self.det_score = score


class MediaPipeFaceAnalysis:
    """
    Drop-in for `FaceAnalysisDIY`: same four methods the Cropper calls, none of the licence.

    Constructed with InsightFace's keyword arguments (`name`, `root`, `providers`) so it can be swapped in
    without touching LivePortrait, and they are ignored.
    """

    def __init__(self, name: str = "buffalo_l", root: str = "", providers=None, **kwargs):
        self._mesh = None
        self._max_faces = 1

    def prepare(self, ctx_id: int = 0, det_size=(512, 512), det_thresh: float = 0.1, **kwargs):
        import mediapipe as mp

        # static_image_mode because every call is an independent image or video frame; the alternative
        # tracks between calls and would carry one frame's face into the next.
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max(1, self._max_faces),
            refine_landmarks=False,
            min_detection_confidence=max(0.1, float(det_thresh)),
        )
        return self

    def warmup(self):
        if self._mesh is None:
            self.prepare()
        # A grey frame rather than black: a pure-black image can fail detection outright, and the point of a
        # warmup is to pay the first-call cost, not to assert anything about it.
        self.get(np.full((512, 512, 3), 128, dtype=np.uint8))

    def get(self, img_bgr: np.ndarray, **kwargs):
        """Return faces ordered as LivePortrait expects, or an empty list when there is no face."""
        if self._mesh is None:
            self.prepare()

        height, width = img_bgr.shape[:2]
        # MediaPipe wants RGB; the Cropper hands us BGR because that is what InsightFace wanted.
        result = self._mesh.process(img_bgr[:, :, ::-1])
        if not result.multi_face_landmarks:
            return []

        faces = []
        for mesh in result.multi_face_landmarks:
            pts = mesh.landmark

            def at(index: int) -> np.ndarray:
                return np.array([pts[index].x * width, pts[index].y * height], dtype=np.float32)

            def eye(pair) -> np.ndarray:
                return (at(pair[0]) + at(pair[1])) / 2.0

            five = np.stack([eye(LEFT_EYE), eye(RIGHT_EYE), at(NOSE_TIP), at(MOUTH_LEFT), at(MOUTH_RIGHT)])

            # A bbox over the whole mesh, which is tighter and steadier than one from five points.
            xs = np.array([p.x for p in pts], dtype=np.float32) * width
            ys = np.array([p.y for p in pts], dtype=np.float32) * height
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            faces.append(_Face(five, bbox, 1.0))

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

    Patched at the module LivePortrait imports the name into, rather than by editing the checkout, so an
    upgrade of LivePortrait does not quietly resurrect the InsightFace path or silently drop this one: if the
    attribute ever stops existing, this raises here instead of failing a paid job later.
    """
    import sys

    if liveportrait_root not in sys.path:
        sys.path.insert(0, liveportrait_root)

    from src.utils import cropper as lp_cropper

    if not hasattr(lp_cropper, "FaceAnalysisDIY"):
        raise RuntimeError(
            "LivePortrait's cropper no longer exposes FaceAnalysisDIY, so the InsightFace replacement cannot "
            "be installed. Check src/utils/cropper.py before running: proceeding would either use "
            "non-commercial InsightFace weights or fail inside a job."
        )
    lp_cropper.FaceAnalysisDIY = MediaPipeFaceAnalysis
