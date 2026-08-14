# The LoRA training worker: ai-toolkit + the Krea 2 training adapter.
#
# A second serverless endpoint, separate from the one that renders. They want
# opposite execution timeouts — seconds for a frame, an hour for a training —
# and one endpoint cannot be both without making a hung render expensive.
#
# No base weights baked in. Krea 2 is already on the network volume for the
# render endpoint, and both attach the same volume, so it is downloaded once.
# That keeps this image small enough to build on a rented pod and push in
# minutes, which matters because there is no local disk to build it on.
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

# The adapter that makes Krea 2 trainable. ~19.7k downloads, the same pattern
# ostris ships for Z-Image and FLUX-schnell. Small enough to bake, and baking it
# means a cold start does not read it over the volume.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ostris/krea2_turbo_training_adapter', local_dir='/adapters/krea2_turbo_training_adapter')"

COPY train-krea2.yaml handler.py /build/

# Fail the build rather than the first training.
RUN test -f /ai-toolkit/run.py || { echo "MISSING: ai-toolkit/run.py"; exit 1; } && \
    test -d /adapters/krea2_turbo_training_adapter || { echo "MISSING: training adapter"; exit 1; } && \
    python -c "import runpod, boto3, yaml, requests" && \
    echo "trainer image OK"

CMD ["python", "-u", "/build/handler.py"]
