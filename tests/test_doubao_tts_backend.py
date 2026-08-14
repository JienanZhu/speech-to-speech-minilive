"""Doubao (Volcengine) cloud TTS backend wiring tests."""

from __future__ import annotations

import struct
from threading import Event

import pytest

from speech_to_speech.backend_registry import TTS_BACKENDS, select_backend
from speech_to_speech.arguments_classes.doubao_tts_arguments import DoubaoTTSHandlerArguments
from speech_to_speech.s2s_pipeline import parse_arguments


def test_doubao_backend_is_registered():
    assert "doubao" in TTS_BACKENDS
    assert TTS_BACKENDS["doubao"].kind == "tts"
    assert TTS_BACKENDS["doubao"].config_prefix == "doubao_tts"


def test_doubao_config_is_normalized_with_prefix_stripped():
    selection = select_backend(
        TTS_BACKENDS,
        "doubao",
        DoubaoTTSHandlerArguments(doubao_tts_speaker="zh_female_test_bigtts"),
    )

    assert selection.config["speaker"] == "zh_female_test_bigtts"
    assert selection.config["resource_id"] == "seed-tts-2.0"
    assert "doubao_tts_speaker" not in selection.config


def test_parser_accepts_doubao_tts_options():
    args = parse_arguments(
        [
            "--tts",
            "doubao",
            "--doubao_tts_speaker",
            "zh_female_test_bigtts",
            "--doubao_tts_speed_ratio",
            "1.2",
        ]
    )

    assert args.tts_backend.name == "doubao"
    assert args.tts_backend.config["speaker"] == "zh_female_test_bigtts"
    assert args.tts_backend.config["speed_ratio"] == 1.2


def test_setup_resolves_credentials_from_environment(monkeypatch):
    from speech_to_speech.TTS.doubao_tts_handler import DoubaoTTSHandler

    monkeypatch.setenv("DOUBAO_TTS_APPID", "app-123")
    monkeypatch.setenv("DOUBAO_TTS_ACCESS_TOKEN", "token-456")

    handler = DoubaoTTSHandler.__new__(DoubaoTTSHandler)
    handler.setup(Event(), resource_id="seed-tts-2.0")

    assert handler.app_id == "app-123"
    assert handler.access_token == "token-456"
    assert handler.resource_id == "seed-tts-2.0"


def test_setup_raises_actionable_error_without_credentials(monkeypatch):
    from speech_to_speech.TTS.doubao_tts_handler import DoubaoTTSHandler

    for key in ("DOUBAO_TTS_APPID", "VOLCENGINE_APP_ID", "DOUBAO_TTS_ACCESS_TOKEN", "VOLCENGINE_ACCESS_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    handler = DoubaoTTSHandler.__new__(DoubaoTTSHandler)
    with pytest.raises(ValueError, match="DOUBAO_TTS_APPID"):
        handler.setup(Event())


def test_message_marshal_unmarshal_roundtrip():
    from speech_to_speech.TTS.doubao_tts_handler import _Event, _Flags, _Message, _MsgType

    original = _Message(
        _MsgType.FullClientRequest,
        _Flags.WithEvent,
        _Event.StartSession,
        session_id="session-1",
        payload=b'{"event": 100}',
    )

    decoded = _Message.unmarshal(original.marshal())

    assert decoded.type == _MsgType.FullClientRequest
    assert decoded.flag == _Flags.WithEvent
    assert decoded.event == _Event.StartSession
    assert decoded.session_id == "session-1"
    assert decoded.payload == b'{"event": 100}'


def test_message_unmarshal_parses_error_code():
    from speech_to_speech.TTS.doubao_tts_handler import _Flags, _Message, _MsgType

    wire = bytes([(1 << 4) | 1, (int(_MsgType.Error) << 4) | int(_Flags.NoSeq), 0x10, 0x00])
    wire += struct.pack(">I", 45000001)
    wire += struct.pack(">I", 4) + b"oops"

    decoded = _Message.unmarshal(wire)

    assert decoded.type == _MsgType.Error
    assert decoded.error_code == 45000001
    assert decoded.payload == b"oops"
