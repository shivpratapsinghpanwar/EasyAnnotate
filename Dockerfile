# Data Factory, as a hosted service.
#
#   docker build -t datafactory .
#   docker run -p 8000:8000 -e DATAFACTORY_API_KEY=change-me \
#              -v datafactory-data:/data datafactory
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
    DATAFACTORY_WORKSPACE=/data \
    PIP_NO_CACHE_DIR=1

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[web] && rm /tmp/*.whl

# A named volume belongs here: uploads are large, and a container filesystem
# is the wrong place to accumulate them.
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
