# MuseTalk 1.5 as a RunPod serverless worker.
# Licence note: MuseTalk code and weights are MIT. We deliberately do NOT download the syncnet
# checkpoint (ByteDance/LatentSync, OpenRAIL++) — it is only needed for training, not inference,
# so the image stays permissively licensed.
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
COPY handler.py /app/MuseTalk/handler.py
CMD ["python", "-u", "handler.py"]
