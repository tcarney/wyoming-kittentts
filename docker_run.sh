#!/usr/bin/env bash
cd /usr/src
exec .venv/bin/python3 -m wyoming_kittentts \
    --uri 'tcp://0.0.0.0:10200' \
    "$@"
