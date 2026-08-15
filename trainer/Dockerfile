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

RUN pip install --no-cache-dir -r /ai-toolkit/requirements.txt && \
    pip install --no-cache-dir runpod boto3 requests pyyaml

# `python3`, never bare `python`. This base image ships python3.11 and pip but
# no `python` on PATH — pip's own upgrade notice in the build log says
# `python3.11 -m pip`. The bare form fails with exit 127, and it failed here,
# three minutes into installing torch, which is a slow way to find out.
#
# The adapter that makes Krea 2 trainable. ~19.7k downloads, the same pattern
# ostris ships for Z-Image and FLUX-schnell. Small enough to bake, and baking it
# means a cold start does not read it over the volume.
RUN python3 -c "\
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
RUN test -f /ai-toolkit/run.py || { echo "MISSING: ai-toolkit/run.py"; exit 1; } && \
    test -d /adapters/krea2_turbo_training_adapter || { echo "MISSING: training adapter"; exit 1; } && \
    python3 -c "import runpod, boto3, yaml, requests" && \
    echo "trainer image OK"

CMD ["python3", "-u", "/build/handler.py"]
