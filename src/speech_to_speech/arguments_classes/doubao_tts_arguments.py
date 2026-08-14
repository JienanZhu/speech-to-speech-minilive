from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DoubaoTTSHandlerArguments:
    doubao_tts_app_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Volcengine speech console AppID for Doubao TTS. Falls back to the "
                "DOUBAO_TTS_APPID or VOLCENGINE_APP_ID environment variable when omitted."
            )
        },
    )
    doubao_tts_access_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Volcengine speech console AccessToken for Doubao TTS. Falls back to the "
                "DOUBAO_TTS_ACCESS_TOKEN or VOLCENGINE_ACCESS_TOKEN environment variable when omitted."
            )
        },
    )
    doubao_tts_resource_id: str = field(
        default="seed-tts-2.0",
        metadata={
            "help": (
                "Volcengine speech resource id selected via X-Api-Resource-Id. "
                "Use 'seed-tts-2.0' for Doubao Seed-TTS 2.0 voices. Default is 'seed-tts-2.0'."
            )
        },
    )
    doubao_tts_speaker: str = field(
        default="zh_female_wenroushunv_uranus_bigtts",
        metadata={
            "help": (
                "Speaker (voice) id enabled in the Volcengine speech console for the selected "
                "resource. Default is 'zh_female_wenroushunv_uranus_bigtts'."
            )
        },
    )
    doubao_tts_speed_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Speech speed multiplier (1.0 = normal). Translated to the API's percent-delta speech_rate."
        },
    )
    doubao_tts_sample_rate: int = field(
        default=24000,
        metadata={"help": "Sample rate requested from the cloud TTS. Audio is resampled to 16 kHz for the pipeline."},
    )
    doubao_tts_blocksize: int = field(
        default=512,
        metadata={"help": "The audio chunk size in samples for streaming output. Default is 512."},
    )
    doubao_tts_timeout: float = field(
        default=30.0,
        metadata={"help": "Timeout in seconds for connecting and receiving from the cloud TTS API."},
    )
