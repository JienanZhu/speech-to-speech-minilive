"""
Doubao (Volcengine) Seed-TTS 2.0 cloud TTS handler.

Streams text to the Volcengine bidirectional streaming TTS WebSocket API and
yields PCM audio back. Synthesis runs entirely in the cloud, so no local model
download is required. Requires an AppID/AccessToken from the Volcengine speech
console with the Seed-TTS resource enabled (X-Api-Resource-Id selects it,
e.g. "seed-tts-2.0").

Wire protocol framing adapted from the MIT-licensed implementation at
https://github.com/Hypnus-Yuan/doubao-speech.
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import uuid
from enum import IntEnum
from threading import Event
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

# Volcengine bidirectional streaming TTS endpoint (domestic, direct connection).
TTS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

# Pipeline playback sample rate; cloud audio is resampled to this.
PIPELINE_SAMPLE_RATE = 16000


class _MsgType(IntEnum):
    Invalid = 0
    FullClientRequest = 0b0001
    AudioOnlyClient = 0b0010
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    FrontEndResultServer = 0b1100
    Error = 0b1111


class _Flags(IntEnum):
    NoSeq = 0b0000
    PositiveSeq = 0b0001
    LastNoSeq = 0b0010
    NegativeSeq = 0b0011
    WithEvent = 0b0100


class _Event(IntEnum):
    None_ = 0
    StartConnection = 1
    FinishConnection = 2
    ConnectionStarted = 50
    ConnectionFailed = 51
    ConnectionFinished = 52
    StartSession = 100
    FinishSession = 102
    SessionStarted = 150
    SessionFinished = 152
    SessionFailed = 153
    TaskRequest = 200
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352
    TTSEnded = 359


# Events that do not carry a session_id field on the wire.
_CONNECTION_EVENTS = frozenset(
    {
        _Event.StartConnection,
        _Event.FinishConnection,
        _Event.ConnectionStarted,
        _Event.ConnectionFailed,
        _Event.ConnectionFinished,
    }
)


class _Message:
    """Unified wire message: 4-byte header + optional fields + payload."""

    def __init__(
        self,
        type_: _MsgType = _MsgType.Invalid,
        flag: _Flags = _Flags.NoSeq,
        event: _Event = _Event.None_,
        session_id: str = "",
        error_code: int = 0,
        payload: bytes = b"",
    ):
        self.type = type_
        self.flag = flag
        self.event = event
        self.session_id = session_id
        self.error_code = error_code
        self.payload = payload

    def marshal(self) -> bytes:
        buf = io.BytesIO()
        buf.write(
            bytes(
                [
                    (1 << 4) | 1,  # version=1, header_size=1 (=> 4 bytes)
                    (int(self.type) << 4) | int(self.flag),
                    0x10,  # serialization=JSON, compression=none
                    0x00,
                ]
            )
        )
        if self.flag == _Flags.WithEvent:
            buf.write(struct.pack(">i", int(self.event)))
            if self.event not in _CONNECTION_EVENTS:
                sid = self.session_id.encode("utf-8")
                buf.write(struct.pack(">I", len(sid)))
                if sid:
                    buf.write(sid)
        if self.type == _MsgType.Error:
            buf.write(struct.pack(">I", self.error_code & 0xFFFFFFFF))
        buf.write(struct.pack(">I", len(self.payload)))
        if self.payload:
            buf.write(self.payload)
        return buf.getvalue()

    @classmethod
    def unmarshal(cls, data: bytes) -> _Message:
        if len(data) < 4:
            raise ValueError(f"frame too short: {len(data)} bytes")
        header_size = (data[0] & 0x0F) * 4
        msg = cls(type_=_MsgType(data[1] >> 4), flag=_Flags(data[1] & 0x0F))

        buf = io.BytesIO(data[header_size:])
        if msg.flag == _Flags.WithEvent:
            msg.event = _Event(struct.unpack(">i", buf.read(4))[0])
            if msg.event not in _CONNECTION_EVENTS:
                sid_size = struct.unpack(">I", buf.read(4))[0]
                if sid_size:
                    msg.session_id = buf.read(sid_size).decode("utf-8", errors="replace")
        if msg.type == _MsgType.Error:
            msg.error_code = struct.unpack(">I", buf.read(4))[0]

        size_bytes = buf.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size:
                msg.payload = buf.read(size)
        return msg


def _resolve_credentials(app_id: Optional[str], access_token: Optional[str]) -> tuple[str, str]:
    app_id = app_id or os.environ.get("DOUBAO_TTS_APPID") or os.environ.get("VOLCENGINE_APP_ID")
    access_token = (
        access_token or os.environ.get("DOUBAO_TTS_ACCESS_TOKEN") or os.environ.get("VOLCENGINE_ACCESS_TOKEN")
    )
    if not app_id or not access_token:
        raise ValueError(
            "Doubao TTS requires credentials: pass --doubao_tts_app_id/--doubao_tts_access_token "
            "or set the DOUBAO_TTS_APPID and DOUBAO_TTS_ACCESS_TOKEN environment variables "
            "(Volcengine speech console -> application management)."
        )
    return app_id, access_token


class DoubaoTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech via the Volcengine Doubao Seed-TTS cloud API.

    Each sentence opens a short-lived WebSocket session, requests raw PCM
    audio, resamples it to the pipeline rate and yields fixed-size int16
    chunks, matching the local TTS handlers' output contract.
    """

    def setup(
        self,
        should_listen: Event,
        app_id: Optional[str] = None,
        access_token: Optional[str] = None,
        resource_id: str = "seed-tts-2.0",
        speaker: str = "zh_female_wenroushunv_uranus_bigtts",
        speed_ratio: float = 1.0,
        sample_rate: int = 24000,
        blocksize: int = 512,
        timeout: float = 30.0,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.app_id, self.access_token = _resolve_credentials(app_id, access_token)
        self.resource_id = resource_id or os.environ.get("DOUBAO_TTS_RESOURCE_ID") or "seed-tts-2.0"
        self.speaker = speaker
        self.speed_ratio = speed_ratio
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.timeout = timeout
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        self._initial_speaker = self.speaker

        # speed_ratio (1.0 multiplier) -> speech_rate (integer percent delta in [-50, 100])
        self.speech_rate = max(-50, min(100, round((self.speed_ratio - 1.0) * 100)))

        logger.info(
            f"Doubao TTS configured: resource={self.resource_id}, speaker={self.speaker}, "
            f"sample_rate={self.sample_rate}, endpoint={TTS_URL}"
        )

    # ------------------------------------------------------------------
    # Wire helpers
    # ------------------------------------------------------------------
    def _send(self, ws, msg: _Message) -> None:
        ws.send(msg.marshal())

    def _recv(self, ws) -> _Message:
        data = ws.recv(timeout=self.timeout)
        if isinstance(data, str):
            raise RuntimeError(f"Doubao TTS: unexpected text frame: {data[:200]!r}")
        msg = _Message.unmarshal(data)
        if msg.type == _MsgType.Error:
            body = msg.payload.decode("utf-8", errors="replace")
            raise RuntimeError(f"Doubao TTS server error [{msg.error_code}]: {body}")
        if msg.event in (_Event.ConnectionFailed, _Event.SessionFailed):
            body = msg.payload.decode("utf-8", errors="replace")
            raise RuntimeError(f"Doubao TTS {msg.event.name}: {body}")
        return msg

    def _build_headers(self) -> dict[str, str]:
        return {
            "X-Api-App-Id": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------
    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        runtime_config = tts_input.runtime_config
        response = tts_input.response

        # Session-level voice overrides map directly to Doubao speaker ids.
        voice: Optional[str] = None
        if response and response.audio and response.audio.output and response.audio.output.voice:
            voice = str(response.audio.output.voice)
        if not voice and runtime_config:
            audio_cfg = runtime_config.session.audio
            audio_output = audio_cfg.output if audio_cfg is not None else None
            voice = str(audio_output.voice) if audio_output is not None and audio_output.voice else None
        if voice:
            self.speaker = voice

        console.print(f"[green]ASSISTANT: {tts_input.text}")
        try:
            yield from self._synthesize(tts_input.text)
        except Exception as e:
            logger.error(f"Doubao TTS synthesis failed: {e}")

    def _synthesize(self, text: str) -> Iterator[np.ndarray]:
        from scipy.signal import resample_poly
        from websockets.sync.client import connect

        gen = self.cancel_scope.generation if self.cancel_scope else None
        headers = self._build_headers()
        session_id = str(uuid.uuid4())

        base_payload = {
            "user": {"uid": headers["X-Api-Request-Id"]},
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": self.speaker,
                "audio_params": {
                    # Raw PCM avoids needing an mp3 decoder locally.
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "speech_rate": self.speech_rate,
                },
            },
        }

        try:
            ws = connect(
                TTS_URL,
                additional_headers=headers,
                max_size=16 * 1024 * 1024,
                open_timeout=self.timeout,
            )
        except Exception as e:
            raise RuntimeError(f"connection failed: {e}") from e

        up = PIPELINE_SAMPLE_RATE // np.gcd(PIPELINE_SAMPLE_RATE, self.sample_rate)
        down = self.sample_rate // np.gcd(PIPELINE_SAMPLE_RATE, self.sample_rate)

        def _to_chunks(audio: np.ndarray) -> Iterator[np.ndarray]:
            """Buffer samples and yield fixed-size int16 chunks."""
            for i in range(0, len(audio), self.blocksize):
                chunk = audio[i : i + self.blocksize]
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

        try:
            # 1. StartConnection
            self._send(
                ws,
                _Message(_MsgType.FullClientRequest, _Flags.WithEvent, _Event.StartConnection, payload=b"{}"),
            )
            msg = self._recv(ws)
            if msg.event != _Event.ConnectionStarted:
                raise RuntimeError(f"expected ConnectionStarted, got {msg.event.name}")

            # 2. StartSession
            start_payload = dict(base_payload, event=int(_Event.StartSession))
            self._send(
                ws,
                _Message(
                    _MsgType.FullClientRequest,
                    _Flags.WithEvent,
                    _Event.StartSession,
                    session_id=session_id,
                    payload=json.dumps(start_payload, ensure_ascii=False).encode("utf-8"),
                ),
            )
            msg = self._recv(ws)
            if msg.event != _Event.SessionStarted:
                raise RuntimeError(f"expected SessionStarted, got {msg.event.name}")

            # 3. TaskRequest carries the text; FinishSession signals no more text.
            task_payload = dict(base_payload, event=int(_Event.TaskRequest))
            task_payload["req_params"] = dict(base_payload["req_params"], text=text)
            self._send(
                ws,
                _Message(
                    _MsgType.FullClientRequest,
                    _Flags.WithEvent,
                    _Event.TaskRequest,
                    session_id=session_id,
                    payload=json.dumps(task_payload, ensure_ascii=False).encode("utf-8"),
                ),
            )
            self._send(
                ws,
                _Message(
                    _MsgType.FullClientRequest, _Flags.WithEvent, _Event.FinishSession, session_id=session_id, payload=b"{}"
                ),
            )

            # 4. Receive audio frames until TTSEnded / SessionFinished.
            leftover = b""
            while True:
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    logger.info("TTS generation cancelled (interruption)")
                    return
                msg = self._recv(ws)
                if msg.type == _MsgType.AudioOnlyServer:
                    leftover += msg.payload
                    usable = len(leftover) - (len(leftover) % 2)  # keep int16 alignment
                    if usable:
                        audio = np.frombuffer(leftover[:usable], dtype=np.int16).astype(np.float32) / 32768.0
                        leftover = leftover[usable:]
                        if self.sample_rate != PIPELINE_SAMPLE_RATE:
                            audio = resample_poly(audio, up=int(up), down=int(down))
                        audio = (audio * 32768).astype(np.int16)
                        yield from _to_chunks(audio)
                elif msg.type == _MsgType.FullServerResponse:
                    if msg.event in (_Event.TTSEnded, _Event.SessionFinished):
                        break
            if leftover:
                logger.warning(f"Doubao TTS: {len(leftover)} trailing PCM bytes dropped")

            # 5. FinishConnection
            self._send(
                ws,
                _Message(_MsgType.FullClientRequest, _Flags.WithEvent, _Event.FinishConnection, payload=b"{}"),
            )
            try:
                msg = self._recv(ws)
                if msg.event != _Event.ConnectionFinished:
                    logger.debug(f"Doubao TTS: expected ConnectionFinished, got {msg.event.name}")
            except TimeoutError:
                logger.warning("Doubao TTS: ConnectionFinished not received in time")
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def on_session_end(self) -> None:
        self.speaker = self._initial_speaker
        logger.debug("Doubao TTS session state reset")
