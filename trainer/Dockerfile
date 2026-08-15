# The LoRA training worker: ai-toolkit + the Krea 2 training adapter.
#
# A second serverless endpoint, separate from the one that renders. They want
# opposite execution timeouts — seconds for a frame, an hour for a training —
# and one endpoint cannot be both without making a hung render expensive.
#
# No base weights baked in. Krea 2 is already on the network volume for the
# render endpoint, and both attach the same volume, so it is downloaded once.
# That keeps this image small enough for RunPod to build in a few minutes.
FROM runpod/base:0.6.2-cuda12.4.1

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pin the commit. ai-toolkit moves fast and a config schema change would
# surface as a training that runs for twenty minutes and then writes nothing.
ARG AI_TOOLKIT_REF=main
RUN git clone --recurse-submodules https://github.com/ostris/ai-toolkit /ai-toolkit && \
    cd /ai-toolkit && git checkout ${AI_TOOLKIT_REF}

# `torchaudio` is not a typo and not ours: ai-toolkit imports it at module load
# (`toolkit/config_modules.py` line 7, reached through jobs/__init__ before any
# training starts) and does not list it in requirements.txt. Without it every
# run dies in the first second with ModuleNotFoundError, after the worker has
# spun up — which reads as "ai-toolkit exited 1" and costs a cold start to find.
#
# Installed in the same resolution pass as the requirements rather than a second
# `pip install`, so the resolver sees torch and torchaudio together and cannot
# pick a pair with mismatched ABIs.
RUN pip install --no-cache-dir -r /ai-toolkit/requirements.txt torchaudio && \
    pip install --no-cache-dir runpod boto3 requests pyyaml

# Then put torch back on CUDA 12.4, which is the one this image is built for.
#
# ai-toolkit's requirements pull torch from PyPI, and the default wheel there is
# now compiled for CUDA 12.8. It installs and imports perfectly; it dies at the
# first `torch.cuda` call, on the worker, after the model has downloaded:
#
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
#
# 12040 is the host driver, not the container. RunPod hands out machines with
# mixed driver ages, so the same image trained on one worker and failed on the
# next — a failure that looks intermittent and is not. Pinning the wheel to
# cu124 means the image runs on any of them.
#
# A second pass rather than a reordering: requirements.txt pins torch itself, so
# installing first would just be overwritten. --force-reinstall is what makes
# this win the last word.
RUN pip install --no-cache-dir --force-reinstall \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124

# `python3.11` everywhere. Not `python`, and not `python3` either.
#
# This base image has three: no bare `python` at all, a system `python3` which
# is Ubuntu 22.04's 3.10, and a 3.11 added through the deadsnakes PPA. `pip`
# belongs to 3.11 and installs into /usr/local/lib/python3.11/dist-packages, so
# `python3` runs the 3.10 that can see none of it.
#
# Both wrong forms failed here, one after the other: `python` with exit 127
# "command not found", then `python3` with ModuleNotFoundError on a package the
# step above had just reported installing. pip's own upgrade notice in the log
# names the right one — `python3.11 -m pip`.
#
# The adapter that makes Krea 2 trainable. ~19.7k downloads, the same pattern
# ostris ships for Z-Image and FLUX-schnell. Small enough to bake, and baking it
# means a cold start does not read it over the volume.
RUN python3.11 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ostris/krea2_turbo_training_adapter', local_dir='/adapters/krea2_turbo_training_adapter')"

# Paths are relative to the REPOSITORY ROOT, not to this file.
#
# The endpoint builds with Dockerfile path `/trainer/Dockerfile` and build
# context `.`, so the context is the repo root and `train-krea2.yaml` alone does
# not resolve — that mismatch is what failed the first build, and it failed with
# an empty log rather than a legible error. If you ever set the build context to
# `/trainer`, drop the `trainer/` prefix here or it breaks the other way.
COPY trainer/train-krea2.yaml trainer/handler.py /build/

# Fail the build rather than the first training.
#
# The ai-toolkit import is the one that matters. Checking `run.py` exists proves
# nothing — the file was there for the run that died, and the failure was three
# imports deep in a dependency it never declared. Importing the job module is
# the same code path a training takes, so a missing package fails here in
# twenty seconds instead of after a GPU cold start.
RUN test -f /ai-toolkit/run.py || { echo "MISSING: ai-toolkit/run.py"; exit 1; } && \
    test -d /adapters/krea2_turbo_training_adapter || { echo "MISSING: training adapter"; exit 1; } && \
    python3.11 -c "import runpod, boto3, yaml, requests" && \
    python3.11 -c "\
import torch, sys; \
v = torch.version.cuda or ''; \
print('torch', torch.__version__, 'cuda', v); \
sys.exit(0 if v.startswith('12.4') else 1)" && \
    cd /ai-toolkit && python3.11 -c "from toolkit.job import get_job" && \
    echo "trainer image OK"

CMD ["python3.11", "-u", "/build/handler.py"]
