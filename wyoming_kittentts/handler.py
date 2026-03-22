import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # int16 = 2 bytes
CHANNELS = 1
SAMPLES_PER_CHUNK = 1024  # ~43ms at 24kHz, matches piper's default

_DEVNULL = open(os.devnull, "w")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries for streaming."""
    sentences = _SENTENCE_END.split(text.strip())
    return [s for s in sentences if s.strip()]


def _synthesize_audio(model, text: str, voice: str) -> bytes:
    """Run inference and convert to int16 PCM bytes (called in executor)."""
    with redirect_stdout(_DEVNULL):
        audio_float = model.generate(text, voice=voice)
    audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    return audio_int16.tobytes()


class KittenTTSEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        model,
        default_voice: str,
        executor: ThreadPoolExecutor,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._wyoming_info_event = wyoming_info.event()
        self._model = model
        self._default_voice = default_voice
        self._executor = executor

    async def run(self) -> None:
        try:
            await super().run()
        except ConnectionResetError:
            pass

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            await self.write_event(self._wyoming_info_event)
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            voice = (
                synthesize.voice.name if synthesize.voice else None
            ) or self._default_voice
            _LOGGER.debug("Synthesizing: text=%r voice=%s", synthesize.text, voice)

            sentences = _split_sentences(synthesize.text)
            if not sentences:
                sentences = [synthesize.text]

            loop = asyncio.get_event_loop()
            bytes_per_chunk = SAMPLES_PER_CHUNK * SAMPLE_WIDTH * CHANNELS

            await self.write_event(
                AudioStart(
                    rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS
                ).event()
            )

            # Stream audio per sentence — reduces time-to-first-audio
            for sentence in sentences:
                audio_bytes = await loop.run_in_executor(
                    self._executor,
                    _synthesize_audio,
                    self._model,
                    sentence,
                    voice,
                )

                for i in range(0, len(audio_bytes), bytes_per_chunk):
                    await self.write_event(
                        AudioChunk(
                            rate=SAMPLE_RATE,
                            width=SAMPLE_WIDTH,
                            channels=CHANNELS,
                            audio=audio_bytes[i : i + bytes_per_chunk],
                        ).event()
                    )

            await self.write_event(AudioStop().event())
            return True

        return True
