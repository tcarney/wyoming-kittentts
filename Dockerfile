FROM debian:bookworm-slim

WORKDIR /usr/src

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml ./
COPY wyoming_kittentts/ ./wyoming_kittentts/

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        git \
        ca-certificates \
    && uv sync --no-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker_run.sh ./

EXPOSE 10200

ENTRYPOINT ["bash", "docker_run.sh"]
