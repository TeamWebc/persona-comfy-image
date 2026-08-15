"""
RunPod serverless handler: train one Krea 2 LoRA and hand back a URL.

The contract is deliberately tiny, because the app already owns everything
around it — dataset curation, the zip, budget reservation, evaluation, the
activation gate. This does one thing:

    in   { dataset_url, trigger_word, steps, lora_rank }
    out  { weights_url, steps, seconds }

`dataset_url` is a zip of numbered images on R2 (`001.png`, `002.png`, ...),
published by `server/media/weights-host.js:publishDatasetZip`. `weights_url` is
the trained `.safetensors` on the same bucket, which the app then fetches,
archives, and copies onto the render volume.

Base weights are read from the network volume rather than baked into this image
— the same volume the render endpoint uses, so Krea 2 is downloaded once and
serves both.
"""

import collections
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import boto3
import requests
import runpod
import yaml

# The volume, mounted by RunPod on every serverless worker.
VOLUME = Path("/runpod-volume")
MODELS = VOLUME / "models"

# ai-toolkit is installed at build time; this is where its config lives.
AI_TOOLKIT = Path("/ai-toolkit")
CONFIG_TEMPLATE = Path("/build/train-krea2.yaml")

# Written to the volume rather than the container: a checkpoint from a run that
# died is worth more than the disk it costs, and the container's is discarded.
OUTPUT_ROOT = VOLUME / "training"


def _bucket():
    """The R2 client. Same bucket the app publishes datasets to."""
    endpoint = os.environ["BUCKET_ENDPOINT_URL"]
    # BUCKET_ENDPOINT_URL carries the bucket as its last path segment so that
    # one variable configures both this and worker-comfyui's uploader.
    base, name = endpoint.rstrip("/").rsplit("/", 1)
    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=base,
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    return client, name


def _fetch_dataset(url: str, into: Path) -> int:
    """Download and unpack the training set. Returns the image count."""
    into.mkdir(parents=True, exist_ok=True)
    archive = into / "dataset.zip"

    with requests.get(url, stream=True, timeout=600) as res:
        res.raise_for_status()
        with open(archive, "wb") as handle:
            for chunk in res.iter_content(chunk_size=1 << 20):
                handle.write(chunk)

    with zipfile.ZipFile(archive) as zf:
        # Flat by construction, but a zip is untrusted input and a member named
        # `../../etc/x` would otherwise escape the directory.
        for member in zf.namelist():
            if member.startswith("/") or ".." in Path(member).parts:
                raise ValueError(f"refusing unsafe zip member: {member}")
        zf.extractall(into)

    archive.unlink()
    images = [p for p in into.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if len(images) < 10:
        raise ValueError(f"only {len(images)} images in the dataset; the trainer needs at least 10")
    return len(images)


def _write_config(job_dir: Path, dataset_dir: Path, trigger_word: str, steps: int, rank: int) -> Path:
    """Fill the ai-toolkit config template for this run."""
    config = yaml.safe_load(CONFIG_TEMPLATE.read_text())
    process = config["config"]["process"][0]

    process["training_folder"] = str(job_dir)
    process["trigger_word"] = trigger_word
    process["network"]["linear"] = rank
    process["network"]["linear_alpha"] = rank
    process["train"]["steps"] = steps
    process["datasets"][0]["folder_path"] = str(dataset_dir)

    # Base weights off the shared volume. Same files the render endpoint loads,
    # so Krea 2 is downloaded once for both.
    process["model"]["name_or_path"] = str(MODELS / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors")
    process["model"]["text_encoder_path"] = str(MODELS / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors")
    process["model"]["vae_path"] = str(MODELS / "vae" / "qwen_image_vae.safetensors")

    out = job_dir / "config.yaml"
    out.write_text(yaml.safe_dump(config))
    return out


def _find_safetensors(root: Path) -> Path:
    """The newest .safetensors under the output dir — ai-toolkit also writes
    intermediate checkpoints, and the last one is the trained LoRA."""
    found = sorted(root.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise RuntimeError("training finished but produced no .safetensors")
    return found[-1]


def handler(job):
    started = time.time()
    payload = job.get("input") or {}

    dataset_url = payload.get("dataset_url")
    trigger_word = payload.get("trigger_word")
    if not dataset_url or not trigger_word:
        return {"error": "dataset_url and trigger_word are both required"}

    steps = int(payload.get("steps") or 1000)
    rank = int(payload.get("lora_rank") or 16)

    # Named after the trigger word, which is already a per-persona hash — so a
    # retry lands in the same place instead of accumulating directories.
    job_dir = OUTPUT_ROOT / trigger_word
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(tempfile.mkdtemp(prefix="dataset-"))

    try:
        count = _fetch_dataset(dataset_url, dataset_dir)
        config_path = _write_config(job_dir, dataset_dir, trigger_word, steps, rank)

        # Streamed AND kept. Both, because they answer different questions.
        #
        # Streaming is what makes a thirty-minute run legible while it happens —
        # silence is indistinguishable from a hang, and the RunPod console log is
        # the only window into it. But `subprocess.run` with no capture throws
        # the output away, so a failure came back as `ai-toolkit exited 1` and
        # nothing else: the app's deadletter row, the LoRA record and the toast
        # all said the same four words, and the traceback existed only in a
        # console log that RunPod purges with the job. That is a GPU cold start
        # spent to learn an exit code, twice.
        #
        # So: print each line as it arrives, and keep the last few. The tail
        # rides home in the error and lands in the deadletter row.
        #
        # `sys.executable` rather than the string "python". This base image
        # ships python3.11 and no bare `python` on PATH — the Dockerfile learned
        # that at build time with an exit 127, and this line would have learned
        # it one GPU cold start into a paid run. It is the interpreter already
        # running us, so it cannot be the wrong one.
        tail = collections.deque(maxlen=40)
        proc = subprocess.Popen(
            [sys.executable, str(AI_TOOLKIT / "run.py"), str(config_path)],
            cwd=str(AI_TOOLKIT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            stripped = line.strip()
            # Progress bars redraw thousands of times and would fill the tail
            # with one step counter, crowding out the traceback underneath it.
            if stripped and not stripped.startswith(("\r", "|")):
                tail.append(stripped)
        code = proc.wait()

        if code != 0:
            return {
                "error": f"ai-toolkit exited {code}: " + " | ".join(list(tail)[-12:]),
                "log_tail": list(tail),
            }

        weights = _find_safetensors(job_dir)

        client, bucket = _bucket()
        key = f"lora/trained/{trigger_word}-r{rank}-s{steps}.safetensors"
        client.upload_file(str(weights), bucket, key)

        public_base = os.environ["BUCKET_PUBLIC_BASE"].rstrip("/")
        return {
            "weights_url": f"{public_base}/{key}",
            "steps": steps,
            "images": count,
            "seconds": round(time.time() - started, 1),
        }
    except Exception as err:  # noqa: BLE001 — the message is the deliverable
        # Returned rather than raised: RunPod reports a raised exception as a
        # bare FAILED with no detail, and the app's probe surfaces this string
        # straight into the deadletter row.
        return {"error": f"{type(err).__name__}: {err}"}
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
