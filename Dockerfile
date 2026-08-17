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

# SAM 3, which is what makes "only her head" possible.
#
# The face pass replaces her face, her hair and its colour, and must leave
# everything below the neck as the bytes it already was. Stock ComfyUI can mask
# a latent — `SetLatentNoiseMask` and `ImageCompositeMasked` are both already
# present on this worker — but nothing in it, and nothing in this repo, can say
# WHERE a head is. Without that the region has to be guessed, and a guess that
# works against a seamless studio ground fails against a room.
#
# ## Why SAM 3, and why a pack rather than the core nodes
#
# SAM 1 and 2 cannot answer this: they are promptable, they segment what you
# point at, and they have no idea what a head is — they would still need
# something to tell them where to look, which is the missing part. SAM 3 takes a
# TEXT prompt, so `"hair, face"` returns the region by name, pixel-accurate,
# open-vocabulary, on any background. That last property is why it beats a face
# detector here: a box around a face either clips the hair or, grown wide enough
# to hold it, takes the shoulders and the background with it, and the
# requirement names hair explicitly.
#
# ComfyUI has supported SAM 3.1 natively since PR #13408, which would mean no
# pack at all — but it needs a core bump, and `COMFYUI_REF` is what every render
# path in this system runs on. A pack touches one graph; a core bump touches all
# of them. Deliberate: this is the smaller blast radius, not the tidier build.
#
# `easy_load_sam3_model` and `easy_sam3_image_segmentation` are the two classes
# the face-pass workflow binds to. The second takes `prompt` as text and returns
# a MASK, which is exactly the shape `SetLatentNoiseMask` wants.
ARG SAM3_REF=main
RUN set -eux; \
    git clone --depth 1 --branch "${SAM3_REF}" \
      https://github.com/yolain/ComfyUI-Easy-Sam3.git \
      /comfyui/custom_nodes/ComfyUI-Easy-Sam3; \
    pip install --no-cache-dir \
      -r /comfyui/custom_nodes/ComfyUI-Easy-Sam3/requirements.txt

# Fail the BUILD if the nodes the face pass binds to are not registered.
#
# Same twenty seconds and the same reasoning as the krea2edit check above, and
# it matters more here: a missing mask node does not degrade the pass, it makes
# the graph unbuildable — and this worker reports an unbuildable graph as a
# COMPLETED job with the reason buried in output.errors.
RUN set -eux; \
    grep -rq 'easy_sam3_image_segmentation' /comfyui/custom_nodes/ComfyUI-Easy-Sam3/ \
      && grep -rq 'easy_load_sam3_model' /comfyui/custom_nodes/ComfyUI-Easy-Sam3/ \
      || { echo "ERROR: ComfyUI-Easy-Sam3 registered neither node — check SAM3_REF"; exit 1; }; \
    echo "sam3 nodes present"

# Point ComfyUI at the SAM 3 weights on the volume.
#
# A symlink rather than a line in `extra_model_paths.yaml`, and the difference
# matters: the yaml in the persona repo is DOCUMENTATION. Nothing copies it into
# this image and no script reads it — the mapping that actually runs ships
# inside `runpod/worker-comfyui`, and it knows about diffusion models, VAEs,
# LoRAs and text encoders because those are the directories ComfyUI has always
# had. `models/sam3` is a directory a custom node invented, so no published
# mapping mentions it and editing our copy would change nothing at all.
#
# Editing the base image's yaml in place would work and is worse: it means
# reproducing every key it already declares, and silently dropping one the day
# RunPod adds it. A symlink asserts one fact and leaves the rest alone.   
#
# Dangling at build time on purpose — /runpod-volume exists only on a running
# worker. A symlink to a path that is not there yet is not an error, it is a
# promise that resolves at mount.
RUN set -eux; \
    mkdir -p /comfyui/models; \
    ln -sfn /runpod-volume/models/sam3 /comfyui/models/sam3; \
    echo "sam3 weights directory linked to the volume"

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
