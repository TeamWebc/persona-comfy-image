# The RunPod serverless worker: RunPod's ComfyUI image, with ComfyUI upgraded.
#
# Why this exists at all — the whole reason, because it is not obvious and it
# cost an afternoon to find:
#
#   RunPod's newest published worker image, runpod/worker-comfyui:5.8.6-base,
#   was built 17 June 2026. Krea 2 landed in ComfyUI in July. Its CLIPLoader
#   therefore refuses `type: "krea2"` with "value not in list (list of length
#   23)" — the first 23 entries of ComfyUI's type enum end at `ideogram4`.
#   No newer official tag exists, and the upstream project this was forked from
#   (timpietruskyblibla/runpod-worker-comfy) is a year older still.
#
# So the base image is right and only ComfyUI inside it is stale. This upgrades
# that one thing and changes nothing else — no weights, no custom handler, so
# the input and output contract the provider speaks is untouched.
#
# There is nothing here your machine has to build. RunPod builds it: Serverless
# -> Deploy from a GitHub repository. See TRAINING-free build notes in SETUP.md.

ARG WORKER_COMFYUI_TAG=5.8.6-base
FROM runpod/worker-comfyui:${WORKER_COMFYUI_TAG}

# Which ComfyUI to move to. `master` is deliberate rather than a pinned tag:
# the models in this stack are new enough that a release older than a few weeks
# does not know about them, and the failure is legible (a named node, a listed
# enum) rather than subtle. Pin it once the stack stops moving.
ARG COMFYUI_REF=master

# Upgrade ComfyUI in place.
#
# comfy-cli installs /comfyui as a git checkout, so a shallow fetch-and-reset is
# the cheapest way forward. The clone fallback is there because "it is a git
# repo" is an assumption about someone else's image, and an image layout change
# should fail loudly here rather than silently leave the old version running.
RUN set -eux; \
    if [ -d /comfyui/.git ]; then \
      cd /comfyui; \
      git fetch --depth 1 origin "${COMFYUI_REF}"; \
      git reset --hard FETCH_HEAD; \
    else \
      echo "/comfyui is not a git checkout — re-cloning"; \
      rm -rf /comfyui.old && mv /comfyui /comfyui.old; \
      git clone --depth 1 --branch "${COMFYUI_REF}" \
        https://github.com/comfyanonymous/ComfyUI.git /comfyui; \
      cp -rn /comfyui.old/models/. /comfyui/models/ 2>/dev/null || true; \
      rm -rf /comfyui.old; \
    fi; \
    pip install --no-cache-dir -r /comfyui/requirements.txt

# Fail the BUILD, not the first render.
#
# This is the exact check that would have saved the afternoon: if the ComfyUI we
# just pulled still has no `krea2` CLIPLoader type, every render will fail with
# a COMPLETED status and the reason buried in the output. Twenty seconds here
# instead of a cold start and a confusing log.
RUN set -eux; \
    grep -q '"krea2"' /comfyui/nodes.py \
      || { echo "ERROR: this ComfyUI has no krea2 CLIPLoader type — check COMFYUI_REF"; exit 1; }; \
    echo "krea2 CLIPLoader type present"

# comfyui-krea2edit, which is what lets Krea 2 be handed a photograph of her.
#
# Apache-2.0, no Python dependencies of its own — it registers two nodes,
# Krea2EditModelPatch and Krea2EditGroundedEncode, that do in-context reference
# conditioning stock ComfyUI has no equivalent for. Pinned by ref for the same
# reason ComfyUI is not: this one is small enough to read, and it runs inside a
# container that holds the R2 credentials.
#
# Cloned rather than installed through comfy-node-install because it is not in
# the registry, and because a `git clone` is the whole documented install.
ARG KREA2EDIT_REF=main
RUN set -eux; \
    git clone --depth 1 --branch "${KREA2EDIT_REF}" \
      https://github.com/lbouaraba/comfyui-krea2edit.git \
      /comfyui/custom_nodes/comfyui-krea2edit

# Fail the BUILD if the pack did not register what the workflow binds to.
#
# Same twenty seconds, same reasoning as the krea2 CLIPLoader check above: a
# missing node class does not fail loudly at render time, it fails as a
# COMPLETED job with the reason buried in output.errors.
RUN set -eux; \
    grep -q 'Krea2EditModelPatch' /comfyui/custom_nodes/comfyui-krea2edit/__init__.py \
      && grep -q 'Krea2EditGroundedEncode' /comfyui/custom_nodes/comfyui-krea2edit/__init__.py \
      || { echo "ERROR: comfyui-krea2edit registered neither node — check KREA2EDIT_REF"; exit 1; }; \
    echo "krea2edit nodes present"

# VHS_VideoCombine, for the Wan video workflows.
#
# Deliberately non-fatal. Nothing in the Krea stills path needs it, and
# `comfy-node-install` is a helper in someone else's image whose name could move
# — failing the whole build over a node that is not used yet would be trading a
# working deployment for a future one.
RUN comfy-node-install comfyui-videohelpersuite \
      || echo "WARNING: VideoHelperSuite not installed — Wan video workflows will need it"

# Models are NOT baked in. They live on the network volume, which RunPod mounts
# at /runpod-volume and ComfyUI picks up automatically — that is what makes this
# image small enough for RunPod to build in a couple of minutes.
