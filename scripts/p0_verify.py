"""P0 验证脚本：文字 -> LLM -> TTS -> 播放 的完整链路。"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

from config import load_config


def generate_reply(user_text: str, config) -> str:
    client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)
    response = client.chat.completions.create(
        model=config.llm.model,
        messages=[
            {"role": "system", "content": "你是真白花音，一位可爱的虚拟歌手。用简短活泼的语气回复，1-2句话。"},
            {"role": "user", "content": user_text},
        ],
    )
    return response.choices[0].message.content


def synthesize_speech(text: str, config) -> bytes:
    response = httpx.post(
        f"{config.tts.base_url}/tts",
        json={
            "text": text,
            "text_lang": "auto",
            "ref_audio_path": config.tts.ref_audio_path,
            "prompt_text": config.tts.ref_text,
            "prompt_lang": config.tts.ref_language,
            "text_split_method": "cut0",
            "media_type": "wav",
            "streaming_mode": False,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.content


def play_audio(wav_bytes: bytes) -> None:
    audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(audio_data, sample_rate)
    sd.wait()


def main() -> None:
    config = load_config()
    print("=" * 52)
    print("    P0 验证 — 花音 TTS 链路测试 (输入 quit 退出)")
    print("=" * 52)

    while True:
        try:
            user_input = input("\n你 > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            print("花音 > (生成中...)", end="\r", flush=True)
            reply = generate_reply(user_input, config)
            print(f"花音 > {reply}")

            print("       (合成语音...)", end="\r", flush=True)
            audio = synthesize_speech(reply, config)
            play_audio(audio)
            print("       ✓ 播放完成          ")

        except KeyboardInterrupt:
            print("\n再见")
            break
        except Exception as e:
            print(f"\n❌ 错误：{e}")


if __name__ == "__main__":
    main()
