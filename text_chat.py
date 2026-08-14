# -*- coding: utf-8 -*-
"""文本聊天客户端：通过键盘与 speech-to-speech 服务端交流。

用法（先用 start-local.ps1 -ServeOnly 启动服务端）：
    py -3.14 -m uv run python text_chat.py

连接 ws://127.0.0.1:8765/v1/realtime，以纯文本模式对话：
会话被设置为 modalities=["text"]，AI 只回文字，不触发 TTS 语音合成。
输入 exit / quit 退出。
"""

import asyncio
import json
import sys

import websockets

REALTIME_URL = "ws://127.0.0.1:8765/v1/realtime"


async def main() -> None:
    try:
        async with websockets.connect(REALTIME_URL, max_size=None) as ws:
            # 切换为纯文本会话：关闭 VAD 打断，只要求文本输出
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "modalities": ["text"],
                            "instructions": "你是一个乐于助人的中文助手，请用简洁的中文回答。",
                            "audio": {
                                "input": {"turn_detection": None},
                            },
                        },
                    }
                )
            )
            print(f"已连接 {REALTIME_URL}（纯文本模式），输入文字开始对话，exit 退出。\n")

            while True:
                # 读取用户输入（放到线程里，避免阻塞事件循环）
                text = await asyncio.to_thread(input, "你> ")
                text = text.strip()
                if not text:
                    continue
                if text.lower() in ("exit", "quit"):
                    print("再见！")
                    return

                # 发送文本消息
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            },
                        }
                    )
                )
                # 触发 AI 生成回复
                await ws.send(json.dumps({"type": "response.create"}))

                # 流式打印回复，直到 response.done
                print("AI> ", end="", flush=True)
                while True:
                    event = json.loads(await ws.recv())
                    etype = event.get("type", "")
                    if etype in ("response.output_text.delta", "response.output_audio_transcript.delta"):
                        print(event.get("delta", ""), end="", flush=True)
                    elif etype == "response.done":
                        print("\n")
                        break
                    elif etype == "error":
                        print(f"\n[服务端错误] {event.get('error', {})}\n")
                        break
    except (ConnectionRefusedError, OSError):
        print(f"无法连接 {REALTIME_URL}，请先运行: .\\start-local.ps1 -ServeOnly")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
