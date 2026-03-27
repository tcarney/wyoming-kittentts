# wyoming-kittentts

A [Wyoming protocol](https://github.com/OHF-Voice/wyoming) server for [KittenTTS](https://github.com/KittenML/KittenTTS) (StyleTTS2-based). Use with Home Assistant or any Wyoming-compatible voice assistant.

## Docker

```bash
# Build
docker build -t wyoming-kittentts .

# Run
docker run -it -p 10200:10200 -v kittentts-data:/data -e HF_HOME=/data wyoming-kittentts \
    --model KittenML/kitten-tts-mini-0.8 --voice Jasper
```

Point HA's Wyoming integration at `<host-ip>:10200`.

> **Note:** Zeroconf auto-discovery does not work from Docker Desktop on macOS (the container runs inside a Linux VM with a different IP). Add the Wyoming integration manually at `<host-ip>:10200`. Zeroconf works correctly when running natively.

## macOS (Native)

Running natively is recommended on macOS for best performance. Requires [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies (uv sync)
script/setup

# Run the server
script/run  --uri tcp://0.0.0.0:10200 \
            --model KittenML/kitten-tts-mini-0.8 \
            --voice Jasper \
            --debug
```

### Install as a Service (launchd)

```bash
script/install
```

Config is stored at `~/.config/kittentts/config`:

```bash
MODEL="KittenML/kitten-tts-mini-0.8"
VOICE="Jasper"
URI="tcp://0.0.0.0:10200"
# DEBUG=true
```

To change options, edit the config and restart:

```bash
$EDITOR ~/.config/kittentts/config
launchctl kickstart -k gui/$UID/com.kittentts.wyoming
```

To uninstall:

```bash
script/uninstall
```

## Voices

Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo

## Models

| Model | Params | Size | HuggingFace ID |
|---|---|---|---|
| kitten-tts-mini | 80M | ~80MB | `KittenML/kitten-tts-mini-0.8` |
| kitten-tts-micro | 40M | ~41MB | `KittenML/kitten-tts-micro-0.8` |
| kitten-tts-nano | 15M | ~56MB | `KittenML/kitten-tts-nano-0.8-fp32` |
| kitten-tts-nano-int8 | 15M | ~25MB | `KittenML/kitten-tts-nano-0.8-int8` |

## Home Assistant App

For use as a Home Assistant app (addon), see [app-kittentts](https://github.com/tcarney/app-kittentts).

## AI Disclosure

This project was developed with assistance from Claude Opus 4.6, based heavily on the [wyoming-piper](https://github.com/rhasspy/wyoming-piper) implementation.

## License

This project is licensed under the [MIT License](LICENSE). See [KittenTTS](https://github.com/KittenML/KittenTTS) for model licensing.
