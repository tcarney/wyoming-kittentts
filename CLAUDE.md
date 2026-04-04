# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wyoming protocol TTS server for [KittenTTS](https://github.com/KittenML/KittenTTS) (StyleTTS2-based, ONNX). Can run standalone (Docker or native) or as part of the [app-kittentts](https://github.com/tcarney/app-kittentts) Home Assistant app. HA connects via the built-in Wyoming Protocol integration — no custom HA integration code needed.

## Development

- **Python version:** >=3.11, <3.13 (required — `misaki` declares `Requires-Python <3.13`)
- **No test suite** — testing is manual via Wyoming protocol connection to HA
- **kittentts is not on PyPI** — resolved via `[tool.uv.sources]` from GitHub (`git+https://github.com/KittenML/KittenTTS.git@0.8.1`)

### Build and Run

```bash
# Native (requires uv: https://docs.astral.sh/uv/)
uv sync
uv run python -m wyoming_kittentts --uri tcp://0.0.0.0:10200 --model KittenML/kitten-tts-mini-0.8 --voice Jasper --debug

# Docker
docker build -t wyoming-kittentts .
docker run -it -p 10200:10200 -v kittentts-data:/data -e HF_HOME=/data wyoming-kittentts \
    --model KittenML/kitten-tts-mini-0.8 --voice Jasper

# macOS launchd service
script/install      # setup + create service with config at ~/.config/kittentts/config.json
script/uninstall    # stop + remove service
```

### Testing

After starting the server, HA's Wyoming integration connects on port 10200. Manually add via Settings > Devices & Services > Add Integration > Wyoming Protocol > `<host-ip>:10200`, or rely on Zeroconf auto-discovery.

## Architecture

### Wyoming Protocol Event Flow

```
Client → Server:  Describe       → Server → Client:  Info (TtsProgram + TtsVoice list)
Client → Server:  Synthesize     → Server → Client:  AudioStart + AudioChunk(s) + AudioStop
```

KittenTTS outputs float32 numpy arrays at 24000 Hz. The handler converts to int16 PCM and streams in ~43ms chunks (1024 samples).

### Key Design Decisions

- **Sentence-level streaming** — text is split on sentence boundaries; audio is sent per sentence to reduce time-to-first-audio
- **Dedicated ThreadPoolExecutor** for inference — single worker prevents CPU cache thrashing; keeps the async event loop free
- **ONNX SessionOptions** applied at startup — `ORT_ENABLE_ALL` graph optimization, auto-detected thread count
- **Warmup inference** on startup — eliminates ONNX JIT cost from first real request
- **Audio conversion runs in the executor** alongside inference (not on the event loop)
- **Zeroconf registration** via `wyoming.zeroconf.HomeAssistantZeroconf` for HA auto-discovery
- **CoreML not supported** — KittenTTS ONNX model has dynamic output shapes incompatible with CoreML EP; CPU provider is used on all platforms

### Package Layout

Flat layout matching wyoming-piper: source at `wyoming_kittentts/`. Entry point: `wyoming_kittentts.__main__:main`.

`kittentts` is listed in `pyproject.toml` dependencies with its GitHub source defined in `[tool.uv.sources]` (not on PyPI).

### Key APIs

```python
# KittenTTS — generates float32 audio at 24000 Hz
from kittentts import KittenTTS
model = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = model.generate("Hello.", voice="Jasper")  # numpy float32 array

# Wyoming — TTS server framework
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.zeroconf import HomeAssistantZeroconf
```

### Voices

Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo

### Models

| Model | Params | Size | HuggingFace ID |
|---|---|---|---|
| kitten-tts-mini | 80M | ~80MB | `KittenML/kitten-tts-mini-0.8` |
| kitten-tts-micro | 40M | ~41MB | `KittenML/kitten-tts-micro-0.8` |
| kitten-tts-nano | 15M | ~56MB | `KittenML/kitten-tts-nano-0.8-fp32` |
| kitten-tts-nano-int8 | 15M | ~25MB | `KittenML/kitten-tts-nano-0.8-int8` |

### Configuration

The app loads a JSON config file at startup (default: `~/.config/kittentts/config.json`). Config values are used as defaults; CLI args override them. Supported keys: `model`, `voice`, `uri`, `threads`, `debug`. Pass `--config <path>` to use a different file.

### macOS launchd Service

- **Install method:** `uv tool install` — installs `wyoming-kittentts` as a standalone tool (no `uv` needed at runtime)
- **Plist generation:** embedded in `script/install` via heredoc (no separate template file)
- **Config file:** `~/.config/kittentts/config.json` — JSON (`model`, `voice`, `uri`, `threads`, `debug`)
- **Logs:** `~/Library/Logs/kittentts/`

## Reference Implementations

- **wyoming-piper**: https://github.com/rhasspy/wyoming-piper — canonical Wyoming server
- **piper addon**: https://github.com/home-assistant/addons/tree/master/piper — canonical HA addon
- **wyoming-mlx-whisper**: https://github.com/basnijholt/wyoming-mlx-whisper — macOS launchd service pattern

## Related Repository

- **app-kittentts**: https://github.com/tcarney/app-kittentts — Home Assistant app that installs this server via pip from GitHub

## Open Questions

- [ ] Verify `espeak-ng` is actually needed inside the container (misaki has an espeak fallback path; may not be exercised for standard English)
- [ ] Consider baking the model into the Docker image for fully offline operation (larger image, but no runtime HuggingFace downloads)
