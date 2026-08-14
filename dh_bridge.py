# DH_live <-> speech-to-speech bridge server
# Serves the DH_live web frontend and bridges /eb_stream to a running
# speech-to-speech realtime server (ws://127.0.0.1:8765/v1/realtime):
# text prompt -> DeepSeek LLM + doubao TTS -> streamed {text, audio, endpoint}.
#
# Prerequisites:
#   1. speech-to-speech server running:  .\start-local.ps1 -ServeOnly
#   2. This bridge:                       uv run python dh_bridge.py   (or .venv\Scripts\python.exe)
#   3. Open http://localhost:8888/static/MiniLive.html  and switch to text mode (keyboard icon)
import asyncio
import base64
import datetime
import ipaddress
import json
import os
import re
import struct
import sys
import threading
import time
import uuid

import uvicorn
import websockets
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

S2S_WS_URL = os.environ.get("S2S_WS_URL", "ws://127.0.0.1:8765/v1/realtime")
# 优先用工作区内可修改的前端副本（dh_static，含静音自动停止），
# 不存在时回退到 DH_live 原版静态目录
_DEFAULT_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dh_static")
if not os.path.isdir(_DEFAULT_STATIC):
    _DEFAULT_STATIC = r"E:\DH_live\web_demo\static"
STATIC_DIR = os.environ.get("DH_LIVE_STATIC", _DEFAULT_STATIC)
AUDIO_SAMPLE_RATE = 16000  # speech-to-speech pipeline output rate
INSTRUCTIONS = "你是一个友好的中文语音助手，用简洁口语化的方式回答，不要太长。"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pcm_to_wav(pcm: bytes, rate: int = AUDIO_SAMPLE_RATE) -> bytes:
    n = len(pcm)
    header = (
        b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", n)
    )
    return header + pcm


def silence_wav(seconds: float = 0.2) -> bytes:
    return pcm_to_wav(b"\x00\x00" * int(AUDIO_SAMPLE_RATE * seconds))


def chunk_json(text: str, audio_b64: str, endpoint: bool) -> str:
    return json.dumps(
        {"text": text, "audio": audio_b64, "endpoint": endpoint},
        ensure_ascii=False,
    ) + "\n"


SENT_END_RE = re.compile(r"[。？！；…!?]")


def _session_update(voice_id: str | None) -> dict:
    """构造 session.update：系统提示词 + 可选的豆包音色覆盖（免重启即时生效）。"""
    session: dict = {"type": "realtime", "instructions": INSTRUCTIONS}
    if voice_id:
        session["audio"] = {"output": {"voice": voice_id}}
    return {"type": "session.update", "session": session}


def webm_to_pcm16_16k(b64: str) -> bytes:
    """解码浏览器 MediaRecorder 录音（webm/opus）为 16kHz 单声道 PCM16。"""
    import io

    import av

    container = av.open(io.BytesIO(base64.b64decode(b64)))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE)
    out = bytearray()
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            out += rf.to_ndarray().tobytes()
    for rf in resampler.resample(None):  # flush
        out += rf.to_ndarray().tobytes()
    container.close()
    return bytes(out)


async def _consume_response(ws, turn_id: str):
    """读取一轮服务端回复，按句流式产出 {text, audio, endpoint} NDJSON 块。

    同时把本轮的「用户语音转文字」和「助手完整回复」写入日志（turn_id 关联）。
    """
    audio_buf = bytearray()
    sent_audio = 0
    text_buf = ""
    sent_chars = 0
    audio_done = False
    full_reply = ""
    deadline = time.monotonic() + 120

    def flush_sentence(sentence: str, endpoint: bool) -> str:
        nonlocal sent_audio
        wav = pcm_to_wav(bytes(audio_buf[sent_audio:]))
        if len(wav) <= 44:
            wav = silence_wav()
        sent_audio = len(audio_buf)
        return chunk_json(sentence, base64.b64encode(wav).decode(), endpoint)

    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.25)
        except asyncio.TimeoutError:
            raw = None
            if time.monotonic() > deadline:
                break
        if raw is not None:
            ev = json.loads(raw)
            etype = ev.get("type")

            if etype == "response.output_audio.delta":
                audio_buf += base64.b64decode(ev.get("delta", ""))

            elif etype in (
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
            ):
                delta = ev.get("delta", "")
                text_buf += delta
                full_reply += delta

            elif etype == "conversation.item.input_audio_transcription.completed":
                # 服务端 STT 最终结果：本轮用户说的话（语音转文字）
                transcript = (ev.get("transcript") or "").strip()
                if transcript:
                    print(f"[{turn_id}] 用户(语音转文字): {transcript}", flush=True)

            elif etype in ("response.audio.done", "response.output_audio.done", "response.done"):
                audio_done = True
                if etype == "response.done":
                    break

            elif etype == "error":
                msg = ev.get("error", {}).get("message", str(ev))
                yield chunk_json(f"[speech-to-speech 出错] {msg}",
                                 base64.b64encode(silence_wav()).decode(), True)
                return

        # 按句切分；非末句先等音频覆盖"已发文字+本句"（约 230ms/字），中途绝不发静音
        while True:
            m = SENT_END_RE.search(text_buf)
            if not m:
                break
            sentence = text_buf[: m.end()].strip()
            if not audio_done:
                need_ms = (sent_chars + len(sentence)) * 230
                have_ms = len(audio_buf) * 1000 // (AUDIO_SAMPLE_RATE * 2)
                if have_ms < need_ms:
                    break  # 音频还没跟上，等下一轮
            text_buf = text_buf[m.end():]
            if sentence:
                sent_chars += len(sentence)
                yield flush_sentence(sentence, False)

    rest_text = text_buf.strip()
    wav = pcm_to_wav(bytes(audio_buf[sent_audio:]))
    if len(wav) <= 44:
        wav = silence_wav()
    # 本轮助手完整回复落日志（含已被按句发出的部分）
    if full_reply.strip():
        print(f"[{turn_id}] 助手: {full_reply.strip()}", flush=True)
    yield chunk_json(rest_text or "（回复结束）",
                     base64.b64encode(wav).decode(), True)


async def s2s_stream(prompt: str, turn_id: str, voice_id: str | None = None):
    """文字模式：把一条文字提问转发给 speech-to-speech，流式返回回复。"""
    try:
        async with websockets.connect(
            S2S_WS_URL, max_size=None, open_timeout=5, close_timeout=2
        ) as ws:
            await ws.send(json.dumps(_session_update(voice_id)))
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            }))
            await ws.send(json.dumps({"type": "response.create"}))

            async for chunk in _consume_response(ws, turn_id):
                yield chunk

    except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
        yield chunk_json(
            "[未连接 speech-to-speech] 请先在 e:\\speech-to-speech 里运行 .\\start-local.ps1 -ServeOnly",
            base64.b64encode(silence_wav()).decode(), True,
        )


async def s2s_stream_audio(audio_b64: str, turn_id: str, voice_id: str | None = None):
    """语音模式：解码录音 → 喂给服务端 VAD→STT→LLM→TTS 全链路，流式返回回复。"""
    try:
        pcm = webm_to_pcm16_16k(audio_b64)
    except Exception as e:
        yield chunk_json(f"[录音解码失败] {e}",
                         base64.b64encode(silence_wav()).decode(), True)
        return

    if len(pcm) < AUDIO_SAMPLE_RATE // 2:  # 不足 0.5 秒
        yield chunk_json("没听清，请再说一遍",
                         base64.b64encode(silence_wav()).decode(), True)
        return

    # 尾部补一段静音，确保 VAD 判定说话结束
    pcm += b"\x00\x00" * int(AUDIO_SAMPLE_RATE * 0.8)

    try:
        async with websockets.connect(
            S2S_WS_URL, max_size=None, open_timeout=5, close_timeout=2
        ) as ws:
            await ws.send(json.dumps(_session_update(voice_id)))
            # 按 4096 字节分片上传录音（服务端会自行切成 512 采样喂 VAD）
            for i in range(0, len(pcm), 4096):
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i:i + 4096]).decode(),
                }))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            got_response = False
            async for chunk in _consume_response(ws, turn_id):
                got_response = True
                yield chunk
            if not got_response:
                yield chunk_json("没听清，请再说一遍",
                                 base64.b64encode(silence_wav()).decode(), True)

    except asyncio.TimeoutError:
        yield chunk_json("没有检测到语音或处理超时，请再说一遍",
                         base64.b64encode(silence_wav()).decode(), True)
    except (ConnectionRefusedError, OSError):
        yield chunk_json(
            "[未连接 speech-to-speech] 请先在 e:\\speech-to-speech 里运行 .\\start-local.ps1 -ServeOnly",
            base64.b64encode(silence_wav()).decode(), True,
        )


@app.post("/eb_stream")
async def stream_response(request: Request):
    data = await request.json()
    input_mode = data.get("input_mode", "text")
    # 前端音色下拉框的值（豆包 speaker id）；空值则沿用服务端当前音色
    voice_id = data.get("voice_id") or None
    # 本轮对话 ID：同一轮的「用户/语音转文字/助手」日志用它关联
    turn_id = uuid.uuid4().hex[:8]
    client_ip = request.client.host if request.client else "?"

    if input_mode == "audio":
        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return JSONResponse(content={"error": "audio is empty"}, status_code=400)
        print(f"[{turn_id}] 收到语音请求 ({len(audio_b64)} bytes base64) ip={client_ip} voice={voice_id}", flush=True)
        return StreamingResponse(s2s_stream_audio(audio_b64, turn_id, voice_id), media_type="application/x-ndjson")

    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(content={"error": "prompt is empty"}, status_code=400)

    print(f"[{turn_id}] 用户(文字): {prompt} ip={client_ip} voice={voice_id}", flush=True)
    return StreamingResponse(s2s_stream(prompt, turn_id, voice_id), media_type="application/x-ndjson")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===== HTTPS 自签证书（局域网麦克风权限）=====
# 浏览器只在安全上下文（HTTPS 或 localhost）开放 getUserMedia，
# 局域网设备走 http://IP 会报"无法访问麦克风"，因此额外提供 HTTPS 端口。
SSL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ssl")
CERT_FILE = os.path.join(SSL_DIR, "dh_cert.pem")
KEY_FILE = os.path.join(SSL_DIR, "dh_key.pem")


def _local_ipv4_addresses() -> list[str]:
    """收集本机 IPv4 地址写进证书 SAN，让局域网设备能用 https://IP 访问。"""
    import socket

    addrs: set[str] = set()
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        addrs.update(info[4][0] for info in infos)
    except OSError:
        pass
    # 通过默认路由探测主网卡 IP（不会真正发包）
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(addrs)


def ensure_self_signed_cert() -> tuple[str, str]:
    """生成自签证书（已存在则直接复用），返回 (certfile, keyfile)。"""
    if os.path.isfile(CERT_FILE) and os.path.isfile(KEY_FILE):
        return CERT_FILE, KEY_FILE

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "dh-bridge")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
    san += [x509.IPAddress(ipaddress.IPv4Address(ip)) for ip in _local_ipv4_addresses()]
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    os.makedirs(SSL_DIR, exist_ok=True)
    with open(KEY_FILE, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"已生成自签证书: {CERT_FILE}（SAN 含 {', '.join(_local_ipv4_addresses()) or '仅本机'}）")
    return CERT_FILE, KEY_FILE


if __name__ == "__main__":
    if not os.path.isdir(STATIC_DIR):
        raise SystemExit(f"找不到 DH_live 静态目录: {STATIC_DIR}（可用环境变量 DH_LIVE_STATIC 修改）")

    # 日志被重定向到文件时先写 UTF-8 BOM：Windows PowerShell 5.1 对无 BOM 文件
    # 默认按 ANSI(GBK) 解读，中文会乱码；有 BOM 后 Get-Content/记事本均正常
    for _stream in (sys.stdout, sys.stderr):
        if not _stream.isatty():
            try:
                _stream.buffer.write(b"\xef\xbb\xbf")
                _stream.flush()
            except (AttributeError, OSError):
                pass

    # HTTP 8888：本机访问；HTTPS 8443：局域网设备访问（麦克风/WebCodecs 需要安全上下文）
    # HTTPS 实例传 log_config=None：日志配置只由下方主实例做一次，
    # 两个实例各配置一遍会让每行日志打两遍
    cert_file, key_file = ensure_self_signed_cert()
    https_server = uvicorn.Server(uvicorn.Config(
        app, host="0.0.0.0", port=8443,
        ssl_certfile=cert_file, ssl_keyfile=key_file, log_config=None,
    ))
    threading.Thread(target=https_server.run, daemon=True).start()
    print("局域网语音访问请用 HTTPS: https://<服务器IP>:8443/static/MiniLive_new.html（首次需点「高级→继续访问」）")
    uvicorn.run(app, host="0.0.0.0", port=8888)
