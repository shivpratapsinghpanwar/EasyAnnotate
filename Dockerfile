# EasyAnnotate, as a hosted service.
#
#   docker build -t easyannotate .
#   docker run -p 8000:8000 -e DATAFACTORY_API_KEY=change-me \
#              -v easyannotate-data:/data easyannotate
#
# Two stages so the runtime image carries no build toolchain. The result is
# small because the core deliberately has no PyTorch dependency -- that is the
# payoff for keeping SAM in an optional extra.

# ---------------------------------------------------------------- build
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
RUN pip install --no-cache-dir build

COPY pyproject.toml README.md LICENSE ./
COPY data_factory ./data_factory
RUN python -m build --wheel --outdir /dist

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim

# opencv-python-headless still wants libgomp for its threaded kernels, but it
# needs neither libGL nor an X stack -- which is the whole reason the package
# defaults to the headless wheel. Installing opencv-python here instead would
# fail at import with "libGL.so.1: cannot open shared object file".
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Never run a service that accepts uploads as root.
RUN useradd --create-home --uid 10001 datafactory

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATAFACTORY_WORKSPACE=/data/workspaces \
    DATAFACTORY_ALLOW_ATTACHED=0 \
    PIP_NO_CACHE_DIR=1

COPY --from=build /dist/*.whl /tmp/dist/
# The obvious `pip install /tmp/*.whl[web]` does not work, and fails in a way
# that looks like pip's fault. `[web]` is a shell *character class* -- match one
# of w, e, b -- so the pattern needs a character after `.whl`, our wheel has
# none, the glob matches nothing, and sh hands pip the literal string
# `/tmp/*.whl[web]`. Resolving the name first and quoting the extra is what
# actually installs the wheel with its extra.
RUN WHEEL="$(ls /tmp/dist/*.whl)" \
 && pip install --no-cache-dir "${WHEEL}[web]" \
 && rm -rf /tmp/dist

# A named volume belongs here: uploads are large, and a container filesystem
# is the wrong place to accumulate them. The workspace is a *subfolder* of the
# volume because the annotation database and project media live beside it, in
# its parent -- with the workspace at /data they landed in /, outside the
# volume, unwritable by this user, and gone on every restart.
RUN mkdir -p /data && chown datafactory:datafactory /data
VOLUME ["/data"]

USER datafactory
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# One worker: jobs live in this process's memory, so a second worker would
# answer polls for jobs it has never heard of. Scale with the job pool inside
# the process, or move jobs to Redis before scaling out.
CMD ["uvicorn", "data_factory.service.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
