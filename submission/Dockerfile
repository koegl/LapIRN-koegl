# Build from the REPOSITORY ROOT, with the two out-of-tree dependencies supplied
# as named build contexts (build.sh does this for you):
#
#   docker build -f submission/Dockerfile -t psmareg_lapirn \
#     --build-context autopet=../autopet-3-submission \
#     --build-context nnunet_model=<Dataset501 model dir> \
#     --build-context totalseg_weights=~/.totalsegmentator/nnunet/results .
#
# Named contexts keep the no-duplication property across repositories: the
# nnU-Net fork and its weights are pulled straight from where they already live,
# so there is still nothing to copy or keep in sync. BuildKit only transfers the
# files a COPY actually references.
#
# Base image choice: torch 2.6 / CUDA 12.4, matching the repo's own pyproject.toml
# so the container runs the same torch the method was developed and trained
# against. CUDA 12.4 is deliberate -- the challenge evaluation machine reports
# CUDA 12.3 / driver 545, and cu12.x builds are minor-version compatible across
# any 12.x driver. A cu13 build would NOT run there.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV TZ=UTC \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

COPY submission/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Fail the BUILD, not the evaluation run, if a dependency dragged in a different
# torch: a silent upgrade to a cu13 wheel makes CUDA unavailable at runtime and
# the container would fall back to CPU (or crash) on the organizers' machine.
RUN python -c "import torch; v=torch.__version__; assert v.startswith('2.6.0'), v; \
    assert torch.version.cuda.startswith('12.'), torch.version.cuda; print('torch', v, 'cuda', torch.version.cuda)"

# --- PET lesion segmentation: the nnU-Net fork from autopet-3-submission -------
# It must be this fork, not PyPI nnunetv2: the checkpoint was trained with the
# custom nnUNetTrainer_PGPSplus, which only exists here (and in its patched
# plans_handler). Installed above the method copy so editing Code/ does not
# invalidate this expensive layer.
COPY --from=autopet nnunetv2 /app/autopet/nnunetv2
COPY --from=autopet pyproject.toml setup.py /app/autopet/
RUN echo "torch==2.6.0" > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt /app/autopet \
    && rm -rf /app/autopet

# main.py is the repo's own in-memory inference wrapper; renamed on the way in
# because a module called `main` on sys.path is too easy to shadow.
COPY --from=autopet main.py /app/autopet_main.py

# Only checkpoint_final.pth is taken -- fold_0 also holds ~17 intermediate
# patch-size checkpoints (~4.5 GB) that the submission has no use for.
# checkpoint_final, not checkpoint_best: the validation split is the held-out set
# the segmentation methods were compared on.
COPY --from=nnunet_model plans.json dataset.json /app/nnunet_model/
COPY --from=nnunet_model fold_0/checkpoint_final.pth /app/nnunet_model/fold_0/

# --- TotalSegmentator: CT organ labels, in its OWN environment -----------------
# It cannot share this one. TotalSegmentator 2.17 imports custom_trainers at
# module level, which needs nnunetv2's `infer_dataset_class` -- absent from the
# autopet fork (2.5.1) installed above, which in turn is the only source of
# nnUNetTrainer_PGPSplus for the PET model. --system-site-packages lets the venv
# reuse torch and friends, so the isolation costs only nnunetv2 + TotalSegmentator.
RUN python -m venv --system-site-packages /opt/tsvenv \
    && /opt/tsvenv/bin/pip install --no-cache-dir \
        nnunetv2==2.8.1 TotalSegmentator==2.17.0

# Only the two datasets `task="total", fast=True, body_seg=True` actually loads:
# the 3 mm total model and the 6 mm body model (~290 MB). The full weights tree
# is 2.8 GB of full-resolution and MRI models this never touches.
COPY --from=totalseg_weights Dataset297_TotalSegmentator_total_3mm_1559subj \
     /app/totalseg_weights/Dataset297_TotalSegmentator_total_3mm_1559subj
COPY --from=totalseg_weights Dataset300_body_6mm_1559subj \
     /app/totalseg_weights/Dataset300_body_6mm_1559subj
ENV TOTALSEG_WEIGHTS_PATH=/app/totalseg_weights \
    TOTALSEG_HOME_DIR=/tmp/totalseg

COPY submission/totalseg_runner.py /app/totalseg_runner.py

# The method lives in Code/ as flat sibling modules (import Functions, import
# my_data, ...), so it is not a package -- putting the directory on PYTHONPATH
# is all that is needed. A bind mount over /app/lapirn/Code shadows this copy
# during development; the baked copy is what the submitted image ships.
COPY Code /app/lapirn/Code
# the crop/restore geometry the TotalSegmentator timing sweep measured
COPY time_totalsegmentator.py /app/lapirn/time_totalsegmentator.py
ENV PYTHONPATH=/app/lapirn/Code:/app/lapirn

COPY submission/weights /app/weights
COPY submission/infer.py /app/infer.py

# The evaluation container runs as an arbitrary uid with no home directory, so
# every library cache has to be redirected somewhere world-writable.
ENV MPLCONFIGDIR=/tmp/mpl \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/tmp/nnunet_results

# COPY preserves the source file modes, and the checkpoint came out of the data
# directory as 0640. The container runs as an arbitrary uid that is in no group
# of the image's files, so anything not world-readable is unreadable -- make the
# read-only payload explicitly world-readable rather than relying on the modes
# the files happened to have on the build host.
RUN chmod -R a+rX /app/weights /app/lapirn /app/infer.py /app/autopet_main.py \
    /app/nnunet_model /app/totalseg_runner.py /app/totalseg_weights

RUN mkdir -p /app/input /app/output && chmod -R 777 /app/output /tmp

# Five positional paths: fixed CT, fixed PET, moving CT, moving PET, output disp.
ENTRYPOINT ["python3", "-u", "/app/infer.py"]
