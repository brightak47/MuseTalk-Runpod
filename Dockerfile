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

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -U openmim huggingface_hub runpod requests boto3 \
 && mim install mmengine \
 && mim install "mmcv==2.0.1" \
 && mim install "mmdet==3.1.0" \
 && mim install "mmpose==1.1.0"

# Weights (~5 GB), fetched through the huggingface_hub Python API: `huggingface-cli` was renamed to
# `hf` in recent releases, so shelling out to it fails with exit 127 depending on the version pip
# resolves. The script verifies every file exists before the build can succeed.
COPY download_weights.py /app/MuseTalk/download_weights.py
RUN python download_weights.py

ENV FFMPEG_PATH=/usr/bin
COPY handler.py /app/MuseTalk/handler.py
CMD ["python", "-u", "handler.py"]
