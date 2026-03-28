import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

import numpy as np
from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

_LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # int16 = 2 bytes
CHANNELS = 1
SAMPLES_PER_CHUNK = 1024  # ~43ms at 24kHz, matches piper's default

_DEVNULL = open(os.devnull, "w")


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

        # Streaming synthesis state
        self._streaming = False
        self._stream_voice: str = default_voice
        self._sbd = SentenceBoundaryDetector()

    async def run(self) -> None:
        try:
            await super().run()
        except ConnectionResetError:
            pass

    async def _send_chunks(self, text: str, voice: str) -> None:
        """Synthesize text and stream audio chunks to the client."""
        loop = asyncio.get_event_loop()
        bytes_per_chunk = SAMPLES_PER_CHUNK * SAMPLE_WIDTH * CHANNELS

        audio_bytes = await loop.run_in_executor(
            self._executor,
            _synthesize_audio,
            self._model,
            text,
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

    async def _synthesize_sentence(self, text: str, voice: str) -> None:
        """Synthesize a single sentence with AudioStart/AudioStop framing."""
        await self.write_event(
            AudioStart(
                rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS
            ).event()
        )
        await self._send_chunks(text, voice)
        await self.write_event(AudioStop().event())

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info_event)
            return True

        if Ping.is_type(event.type):
            ping = Ping.from_event(event)
            await self.write_event(Pong(text=ping.text).event())
            return True

        if Synthesize.is_type(event.type):
            if self._streaming:
                # Ignore — only sent for compatibility during streaming sessions.
                # Text arrives via SynthesizeChunk events.
                _LOGGER.debug("Ignoring synthesize during streaming session")
                return True

            synthesize = Synthesize.from_event(event)
            voice = (
                synthesize.voice.name if synthesize.voice else None
            ) or self._default_voice
            _LOGGER.debug("Synthesizing: text=%r voice=%s", synthesize.text, voice)

            try:
                sbd = SentenceBoundaryDetector()
                start_sent = False

                for sentence in sbd.add_chunk(synthesize.text):
                    await self._synthesize_sentence(sentence, voice)
                    start_sent = True

                remaining = sbd.finish()
                if remaining:
                    await self._synthesize_sentence(remaining, voice)
                elif not start_sent:
                    # No sentences detected — send empty audio frame
                    await self.write_event(
                        AudioStart(
                            rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS
                        ).event()
                    )
                    await self.write_event(AudioStop().event())
            except Exception as exc:
                _LOGGER.exception("Synthesis failed: text=%r voice=%s", synthesize.text, voice)
                await self.write_event(
                    Error(text=f"Synthesis failed: {exc}", code="tts-error").event()
                )

            return True

        # --- Streaming synthesis ---

        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._stream_voice = (
                start.voice.name if start.voice else None
            ) or self._default_voice
            self._sbd = SentenceBoundaryDetector()
            self._streaming = True
            _LOGGER.debug("Streaming synthesis started: voice=%s", self._stream_voice)
            return True

        if SynthesizeChunk.is_type(event.type):
            if not self._streaming:
                _LOGGER.warning("Received synthesize-chunk outside streaming session")
                return True

            chunk = SynthesizeChunk.from_event(event)

            try:
                for sentence in self._sbd.add_chunk(chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %r", sentence)
                    await self._synthesize_sentence(sentence, self._stream_voice)
            except Exception as exc:
                _LOGGER.exception("Streaming synthesis failed")
                self._streaming = False
                self._sbd = SentenceBoundaryDetector()
                await self.write_event(
                    Error(text=f"Synthesis failed: {exc}", code="tts-error").event()
                )

            return True

        if SynthesizeStop.is_type(event.type):
            if not self._streaming:
                _LOGGER.warning("Received synthesize-stop outside streaming session")
                return True

            _LOGGER.debug("Streaming synthesis stopping, flushing remaining text")
            try:
                remaining = self._sbd.finish()
                if remaining:
                    await self._synthesize_sentence(remaining, self._stream_voice)

                await self.write_event(SynthesizeStopped().event())
            except Exception as exc:
                _LOGGER.exception("Streaming synthesis failed during flush")
                await self.write_event(
                    Error(text=f"Synthesis failed: {exc}", code="tts-error").event()
                )
            finally:
                self._streaming = False
                self._sbd = SentenceBoundaryDetector()

            return True

        return True
