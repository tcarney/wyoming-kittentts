import argparse
import asyncio
import json
import logging
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import onnxruntime as ort
from kittentts import KittenTTS
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer
from wyoming.zeroconf import HomeAssistantZeroconf

from .handler import KittenTTSEventHandler, _synthesize_audio

VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]

_DEFAULT_CONFIG = os.path.join(os.environ.get("HOME", "~"), ".config/kittentts/config.json")

_LOGGER = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    """Load JSON config file, returning a dict of settings."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


async def _async_main():
    # First pass: extract --config so we can use it as defaults for everything else
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=_DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()

    file_cfg = _load_config(pre_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=_DEFAULT_CONFIG,
                        help="Path to config file (default: ~/.config/kittentts/config)")
    parser.add_argument("--uri", default=file_cfg.get("uri", "tcp://0.0.0.0:10200"))
    parser.add_argument("--model", default=file_cfg.get("model", "KittenML/kitten-tts-mini-0.8"))
    parser.add_argument("--voice", default=file_cfg.get("voice", "Jasper"))
    parser.add_argument(
        "--threads", type=int,
        default=file_cfg.get("threads", 0),
        help="ONNX intra-op threads (0 = auto-detect CPU count)",
    )
    parser.add_argument("--debug", action="store_true",
                        default=file_cfg.get("debug", False))
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if file_cfg:
        _LOGGER.info("Loaded config from %s", pre_args.config)
    else:
        _LOGGER.debug("No config file found at %s", pre_args.config)

    # Suppress verbose HTTP wire logging
    http_level = logging.INFO if args.debug else logging.WARNING
    logging.getLogger("httpcore").setLevel(http_level)
    logging.getLogger("httpx").setLevel(http_level)

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="kittentts",
                attribution=Attribution(
                    name="KittenML",
                    url="https://github.com/KittenML/KittenTTS",
                ),
                installed=True,
                description="KittenTTS local TTS",
                version="0.8.1",
                supports_synthesize_streaming=True,
                voices=[
                    TtsVoice(
                        name=v,
                        attribution=Attribution(
                            name="KittenML",
                            url="https://github.com/KittenML/KittenTTS",
                        ),
                        installed=True,
                        description=v,
                        version="0.8.1",
                        languages=["en"],
                    )
                    for v in VOICES
                ],
            )
        ]
    )

    # Configure ONNX Runtime for lower latency
    ort.set_default_logger_severity(3)  # suppress ORT warnings
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if args.threads > 0:
        sess_options.intra_op_num_threads = args.threads
    # else: ONNX Runtime auto-detects (uses all cores)

    _LOGGER.info("Loading KittenTTS model: %s", args.model)
    model = KittenTTS(args.model)

    # Apply optimized session options to the ONNX model
    model.model.session = ort.InferenceSession(
        model.model.model_path, sess_options=sess_options
    )

    # Warmup: run a dummy inference to JIT-compile the ONNX graph
    _LOGGER.info("Warming up model...")
    _synthesize_audio(model, "Hello.", args.voice)
    _LOGGER.info("Model ready. Starting server on %s", args.uri)

    # Dedicated executor for inference — avoids default pool contention
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    # Parse port from URI for Zeroconf registration
    uri_parts = args.uri.rsplit(":", 1)
    port = int(uri_parts[-1]) if len(uri_parts) > 1 else 10200

    zeroconf = HomeAssistantZeroconf(port=port, name="kittentts")
    await zeroconf.register_server()
    _LOGGER.info("Zeroconf discovery registered on port %d", port)

    server = AsyncServer.from_uri(args.uri)
    handler_factory = partial(
        KittenTTSEventHandler, wyoming_info, model, args.voice, executor
    )

    server_task = asyncio.create_task(server.run(handler_factory))
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, server_task.cancel)
    loop.add_signal_handler(signal.SIGTERM, server_task.cancel)

    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("Server stopped")


def main():
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        _LOGGER.info("Stopping")


if __name__ == "__main__":
    main()
