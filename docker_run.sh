#!/usr/bin/env bash
cd /usr/src
exec uv run python -m wyoming_kittentts \
    --uri 'tcp://0.0.0.0:10200' \
    "$@"
