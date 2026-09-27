# MuseTalk lip-sync worker (RunPod serverless)

Wraps [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) (MIT) as a RunPod serverless endpoint
for Adpence / Intelligenfy / Videopost AI.

## Request

```json
{
  "input": {
    "audio_url": "https://.../narration.wav",
    "video_url": "https://.../idle_loop.mp4",
    "fps": 25,
    "output_key": "lessons/42/slice_000.mp4",
    "project": "intelligenfy"
  }
}
```

`image_url` works instead of `video_url` for a still. `*_b64` variants are accepted for all inputs.

## Response

`video_url` when R2/S3 is configured, else `video_path` on the network volume, else `video_b64`
for small results. Always includes `seconds` and `size_bytes`, plus `project` when supplied.

## Endpoint environment

| Variable | Purpose |
|---|---|
| `R2_ACCOUNT_ID`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | upload results to Cloudflare R2 |
| `R2_PUBLIC_BASE_URL` | return a public URL instead of a 7-day presigned one |
| `S3_ENDPOINT_URL`, `S3_BUCKET`, `AWS_*` | any other S3-compatible store |

## Licence notes

MuseTalk code and weights are MIT; sd-vae-ft-mse (MIT), whisper-tiny (Apache-2.0) and DWPose
(Apache-2.0) are all permissive. The image deliberately omits MuseTalk's `syncnet` checkpoint
(ByteDance/LatentSync, OpenRAIL++) because it is only used for training.

One loose end: `face-parse-bisent` (`79999_iter.pth`) is fetched from a Google Drive link with no
stated licence. It is required for v15 face blending. Have this reviewed before commercial launch.
