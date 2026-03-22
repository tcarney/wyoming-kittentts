FROM debian:bookworm-slim

WORKDIR /usr/src

COPY pyproject.toml ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        espeak-ng \
        git \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        "kittentts @ git+https://github.com/KittenML/KittenTTS.git@0.8.1" \
    && .venv/bin/pip3 install --no-cache-dir -e '.' \
    \
    && rm -rf /var/lib/apt/lists/*

COPY wyoming_kittentts/ ./wyoming_kittentts/
COPY docker_run.sh ./

EXPOSE 10200

ENTRYPOINT ["bash", "docker_run.sh"]
