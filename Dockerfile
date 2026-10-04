# MuseTalk 1.5 and LivePortrait as one RunPod serverless worker.
#
# MuseTalk syncs a mouth to audio; LivePortrait gives a still portrait head motion first. Together they
# make a presentation video, and keeping them in one image means a job pays one cold start on ~15 GB
# rather than two, with no intermediate clip crossing the network.
#
# Licences: MuseTalk and LivePortrait are both MIT, and so are LivePortrait's own weights.
#
# Two sets of weights are deliberately absent:
#   * MuseTalk's syncnet checkpoint (ByteDance/LatentSync, OpenRAIL++) is only used for training.
#   * InsightFace's pretrained models, which LivePortrait's Cropper loads by default, are released for
#     non-commercial research only. face_detect.py replaces them with YuNet (OpenCV Zoo, MIT), and
#     verify_liveportrait.py fails the build if they ever reappear on disk -- a rebuild that pulled them
#     back in would be an infringement rather than a bug.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
WORKDIR /app/MuseTalk

# huggingface_hub is pinned below 1.0 on purpose: MuseTalk pins a transformers release that
# requires <1.0, and 2.0.0 also renamed the CLI. download_weights.py uses the Python API, so the
# pin costs us nothing.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -U openmim runpod requests boto3 "huggingface_hub>=0.23,<1.0" \
 && mim install mmengine \
 && mim install "mmcv==2.0.1" \
 && mim install "mmdet==3.1.0" \
 && mim install "mmpose==1.1.0" \
 && python -c "import transformers, huggingface_hub; print('transformers', transformers.__version__, '| hub', huggingface_hub.__version__)"

# Weights (~5 GB). The script verifies every file exists, so a missing weight fails the build
# rather than the first inference request.
COPY download_weights.py /app/MuseTalk/download_weights.py
RUN python download_weights.py

# The PyTorch base image ships a conda ffmpeg built without libx264, and it shadows the apt one on
# PATH. MuseTalk shells out to plain `ffmpeg`, so it picked up the crippled build and died with
# "Unrecognized option 'crf'" after a full, successful inference pass. Remove the conda binaries and
# prove the remaining ffmpeg can actually encode H.264.
RUN rm -f /opt/conda/bin/ffmpeg /opt/conda/bin/ffprobe  && ffmpeg -hide_banner -h encoder=libx264 > /dev/null  && ffmpeg -hide_banner -version | head -1

ENV FFMPEG_PATH=/usr/bin

# LivePortrait (MIT) animates a still portrait so the presenter has head motion before MuseTalk syncs the
# mouth. One image rather than two endpoints: a presentation job then pays a single cold start on a ~15 GB
# image instead of two, and the intermediate clip never leaves the worker.
RUN git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git /app/LivePortrait

# Only requirements_base.txt, deliberately. The full requirements.txt adds transformers==4.38.0, which
# would repin the version MuseTalk needs -- the same shape of fault as capping huggingface_hub did. gradio
#
# No mediapipe either. It was the first attempt at replacing InsightFace and it pulls in TensorFlow,
# which then failed against this image's numpy -- MuseTalk, LivePortrait and TensorFlow each want a
# different one. YuNet runs through the OpenCV already installed here and adds no dependency at all.
# is dropped because the demo UI is never started here, and onnxruntime is the CPU build: upstream pins
# onnxruntime-gpu==1.18, which expects CUDA 12 while this image is 11.7, and landmark.onnx runs once per
# portrait so the CPU cost is paid once per avatar. liveportrait_runner forces that choice explicitly.
RUN grep -vE '^(gradio|onnxruntime)' /app/LivePortrait/requirements_base.txt > /tmp/lp_req.txt  && pip install --no-cache-dir -r /tmp/lp_req.txt  && pip install --no-cache-dir onnxruntime  && python -c "import transformers, huggingface_hub; print('after LivePortrait deps: transformers', transformers.__version__, '| hub', huggingface_hub.__version__)"

COPY face_detect.py /app/MuseTalk/face_detect.py
COPY liveportrait_runner.py /app/MuseTalk/liveportrait_runner.py

# Proven at build time: the weights we may use are present, the ones we may not are absent, the YuNet
# substitution takes hold, and LivePortrait imports. The licence half is the point -- a rebuild that
# quietly pulled InsightFace back in would be an infringement, not a bug, so it fails the build.
COPY verify_liveportrait.py /app/MuseTalk/verify_liveportrait.py
RUN python verify_liveportrait.py

COPY handler.py /app/MuseTalk/handler.py
CMD ["python", "-u", "handler.py"]
