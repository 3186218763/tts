"""AI 花音 CLI 命令行界面。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from dialogue.audio_player import AudioPlayer
from dialogue.conversation import Conversation
from dialogue.llm_client import LLMClient
from dialogue.orchestrator import Orchestrator
from dialogue.tts_client import TTSClient


async def main() -> None:
    config = load_config()

    llm = LLMClient(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.model,
        protocol=config.llm.protocol,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        frequency_penalty=config.llm.frequency_penalty,
    )
    tts = TTSClient(
        base_url=config.tts.base_url,
        ref_audio_path=config.tts.ref_audio_path,
        ref_text=config.tts.ref_text,
        ref_language=config.tts.ref_language,
        text_language=config.tts.text_language,
        top_k=config.tts.top_k,
        top_p=config.tts.top_p,
        temperature=config.tts.temperature,
        repetition_penalty=config.tts.repetition_penalty,
        speed_factor=config.tts.speed_factor,
        seed=config.tts.seed,
        text_split_method=config.tts.text_split_method,
    )
    try:
        await tts.check_available()
    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return
    player = AudioPlayer()
    orchestrator = Orchestrator(
        llm,
        tts,
        player,
        max_chars=config.max_sentence_chars,
        min_chars=config.min_sentence_chars,
    )
    conversation = Conversation(
        recent_turns=config.max_turns,
        summary_trigger_turns=config.summary_trigger_turns,
        summary_trigger_chars=config.summary_trigger_chars,
        summary_max_chars=config.summary_max_chars,
    )

    print("=" * 52)
    print("       🌸 AI 花音 — 对话模式 (Ctrl+C 退出)")
    print("=" * 52)

    while True:
        try:
            user_input = (await asyncio.to_thread(input, "\n你 > ")).strip()
            if not user_input:
                continue

            print("花音 > ", end="", flush=True)
            async for sentence in orchestrator.chat(user_input, conversation):
                print(sentence, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\n\n再见～ 🌸")
            break
        except Exception as e:
            print(f"\n❌ 出错了：{e}")


if __name__ == "__main__":
    asyncio.run(main())
