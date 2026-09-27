# MuseTalk 1.5 as a RunPod serverless worker.
# Licence note: MuseTalk code and weights are MIT. We deliberately do NOT download the syncnet
# checkpoint (ByteDance/LatentSync, OpenRAIL++) — it is only needed for training, not inference,
# so the image stays MIT-clean apart from face-parse-bisent (see README).
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
WORKDIR /app/MuseTalk

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -U openmim "huggingface_hub[cli]" gdown runpod requests boto3 \
 && mim install mmengine \
 && mim install "mmcv==2.0.1" \
 && mim install "mmdet==3.1.0" \
 && mim install "mmpose==1.1.0"

# Weights (~5 GB). syncnet is intentionally omitted: training-only and a different licence.
RUN mkdir -p models/musetalk models/musetalkV15 models/dwpose models/face-parse-bisent models/sd-vae models/whisper \
 && huggingface-cli download TMElyralab/MuseTalk --local-dir models --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" \
 && huggingface-cli download TMElyralab/MuseTalk --local-dir models --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth" \
 && huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae --include "config.json" "diffusion_pytorch_model.bin" \
 && huggingface-cli download openai/whisper-tiny --local-dir models/whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json" \
 && huggingface-cli download yzd-v/DWPose --local-dir models/dwpose --include "dw-ll_ucoco_384.pth" \
 && gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth \
 && curl -fsSL https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth

ENV FFMPEG_PATH=/usr/bin
COPY handler.py /app/MuseTalk/handler.py
CMD ["python", "-u", "handler.py"]
