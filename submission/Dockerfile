# Build from the REPOSITORY ROOT so that Code/ is inside the build context:
#   docker build -f submission/Dockerfile -t psmareg_lapirn .
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

# The method lives in Code/ as flat sibling modules (import Functions, import
# my_data, ...), so it is not a package -- putting the directory on PYTHONPATH
# is all that is needed. A bind mount over /app/lapirn/Code shadows this copy
# during development; the baked copy is what the submitted image ships.
COPY Code /app/lapirn/Code
ENV PYTHONPATH=/app/lapirn/Code

COPY submission/weights /app/weights
COPY submission/infer.py /app/infer.py

# The evaluation container runs as an arbitrary uid with no home directory, so
# every library cache has to be redirected somewhere world-writable.
ENV MPLCONFIGDIR=/tmp/mpl \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp

# COPY preserves the source file modes, and the checkpoint came out of the data
# directory as 0640. The container runs as an arbitrary uid that is in no group
# of the image's files, so anything not world-readable is unreadable -- make the
# read-only payload explicitly world-readable rather than relying on the modes
# the files happened to have on the build host.
RUN chmod -R a+rX /app/weights /app/lapirn/Code /app/infer.py

RUN mkdir -p /app/input /app/output && chmod -R 777 /app/output /tmp

# Five positional paths: fixed CT, fixed PET, moving CT, moving PET, output disp.
ENTRYPOINT ["python3", "-u", "/app/infer.py"]
