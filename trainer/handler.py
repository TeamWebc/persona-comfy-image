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

# ---------------------------------------------------------------------------
# Everything HuggingFace writes goes on the volume, not in the container.
#
# `_ensure_base_weight` below fetches the two checkpoints we know about, and it
# writes them to the volume deliberately. But ai-toolkit ALSO downloads on its
# own account — the Qwen text encoder's tokenizer and companion files — and
# those go through `hf_hub_download`'s default cache, which lives on the
# container's overlay filesystem. That overlay is a few gigabytes; the volume is
# 150. So a run with 73GB free on the volume still died at
#
#   _hf_hub_download_to_cache_dir -> OSError: [Errno 122] Disk quota exceeded
#   Reconstructing: 37% | 3.24GB / 8.88GB
#
# having filled a disk nobody meant to be using. Set before any import that
# might pull in huggingface_hub, because the cache location is read at import.
#
# HF_HOME moves the whole tree (hub cache, tokenizers, datasets); HF_HUB_CACHE
# is set too because a library that reads only the more specific variable would
# otherwise still land in the container.
# ---------------------------------------------------------------------------
# The Dockerfile already sets HF_HOME to /runpod-volume/.cache/huggingface, and
# `setdefault` leaves it alone — this is the fallback for a worker started
# without it. The two below are DERIVED from whatever HF_HOME ends up being
# rather than written out again: hardcoding a second absolute path split the
# cache across two directories and re-downloaded what was already there.
os.environ.setdefault("HF_HOME", str(VOLUME / ".cache" / "huggingface"))
_HF_HOME = Path(os.environ["HF_HOME"])
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME / "transformers"))
# Xet's reconstruction path is what produced both the "Background writer
# channel closed" crash and the partial-write above; plain HTTP is slower and
# survives.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
for _cache in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
    Path(os.environ[_cache]).mkdir(parents=True, exist_ok=True)

# The training checkpoints, which are NOT the rendering ones.
#
# The render endpoint loads `*_fp8_scaled`, and those carry a `weight_scale`
# tensor beside every quantised weight. ComfyUI expects that; ai-toolkit's
# SingleStreamDiT does not, and `load_state_dict(strict=True)` rejects all 232
# of them:
#
#   Unexpected key(s) in state_dict: "blocks.0.attn.wq.weight_scale", ...
#
# So training reads the plain bf16 variants of the same model. Bigger on disk
# and bigger to load; `quantize: true` in the config brings them back down once
# they are in memory. Downloaded on first use rather than baked into the image,
# because they are ~34GB and both endpoints share the volume they land on.
TRAIN_REPO = "Comfy-Org/Krea-2"
TRAIN_FILES = {
    # RAW, not turbo — see the long note beside `arch` in train-krea2.yaml.
    #
    # Training against the turbo distillation requires ostris' training adapter,
    # and that adapter reaches eight fewer modules than a LoRA needs to carry
    # skin texture and hair tone. The raw checkpoint trains without an adapter
    # and reaches all of them. Inference is unchanged: the renderer still loads
    # krea2_turbo_fp8_scaled, and a LoRA transfers between a model and its own
    # distillation. The turbo bf16 stays on the volume — reverting is this one
    # line, not another 26GB download.
    "name_or_path": "diffusion_models/krea2_raw_bf16.safetensors",
    "text_encoder_path": "text_encoders/qwen3vl_4b_bf16.safetensors",
    # Not quantised in the first place, so the render copy is the training copy.
    "vae_path": "vae/qwen_image_vae.safetensors",
}

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


def _remote_size(rel: str) -> int:
    """How big the checkpoint is, without downloading it. 0 if HF won't say."""
    url = f"https://huggingface.co/{TRAIN_REPO}/resolve/main/{rel}"
    try:
        res = requests.head(url, allow_redirects=True, timeout=30)
        return int(res.headers.get("content-length") or 0)
    except Exception:  # noqa: BLE001 — a missing preflight must not block the run
        return 0


def _ensure_base_weight(rel: str) -> Path:
    """
    The training checkpoint, fetched to the volume the first time it is wanted.

    Idempotent and shared: both endpoints mount the same volume, so this is
    downloaded once ever, not once per run. Written to a temp name and renamed,
    because two workers starting together would otherwise both be part-way
    through the same 26GB file and each would see the other's half as complete.
    """
    target = MODELS / rel
    if target.exists() and target.stat().st_size > 0:
        return target

    # Cache location and the Xet switch are both set at module scope, before
    # anything can import huggingface_hub — see the block under VOLUME.
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)

    # Fail in seconds rather than twenty minutes.
    #
    # `Disk quota exceeded` arrived partway through a 26GB download, on a GPU
    # that had been paid for since the cold start. The volume is a known size
    # and the file has a known size, so the answer is available before the
    # first byte moves — and the operator gets told how much to add rather
    # than being handed an errno.
    need = _remote_size(rel)
    free = shutil.disk_usage(MODELS).free
    if need and free < need + (2 << 30):  # 2GB of headroom for the trainer itself
        raise RuntimeError(
            f"not enough room on the volume for {rel}: needs {need / 1e9:.1f} GB "
            f"plus headroom, {free / 1e9:.1f} GB free. Grow the network volume by "
            f"at least {((need + (2 << 30) - free) / 1e9):.0f} GB and retry."
        )

    # A previous run that died mid-download leaves an incomplete blob behind,
    # and it counts against the quota while being worth nothing.
    for stale in MODELS.glob(".cache/huggingface/**/*.incomplete"):
        stale.unlink(missing_ok=True)

    print(f"fetching training weight {rel} ({need / 1e9:.1f} GB, first run only)", flush=True)

    # `local_dir` writes the real file straight to its final path.
    #
    # Without it `hf_hub_download` lands the file in its own cache and returns
    # that path, leaving us to copy it across — which for a 26GB checkpoint
    # means 52GB on a volume that already holds ~19GB of render models, and
    # `OSError: [Errno 122] Disk quota exceeded` partway through the copy. HF
    # writes to `local_dir/<filename>`, which is exactly where MODELS wants it,
    # so there is no second copy to run out of room for.
    hf_hub_download(repo_id=TRAIN_REPO, filename=rel, local_dir=str(MODELS))

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"download finished but {target} is missing or empty")

    print(f"  -> {target} ({target.stat().st_size / 1e9:.1f} GB)", flush=True)
    return target


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

    # Base weights off the shared volume — the bf16 variants, not the
    # fp8_scaled ones the renderer uses. See TRAIN_FILES for why.
    for field, rel in TRAIN_FILES.items():
        process["model"][field] = str(_ensure_base_weight(rel))

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
        # console log that RunPod purges with the job. That is a GPU-hour spent
        # to learn an exit code.
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
