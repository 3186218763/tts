# LLM 客户端 Anthropic 原生协议支持 实现计划

> **面向 AI 代理的工作者:** 必需子技能:使用 superpowers:subagent-driven-development(推荐)或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框(`- [ ]`)语法来跟踪进度。

**目标:** 为 `LLMClient` 新增 Anthropic 原生协议(`/v1/messages`)支持,通过配置 `protocol: anthropic` 切换,OpenAI 兼容路径完全保留。

**架构:** `LLMClient` 构造时按 `protocol` 分派内部客户端——`openai` 用现有 `AsyncOpenAI`,`anthropic` 用项目已有的 `httpx.AsyncClient`(零新增依赖)。两个公开方法 `stream_chat()` / `summarize_chat()` 签名不变,上游 orchestrator/memory 零改动。请求 URL 按 Anthropic SDK 风格拼接:`https://opencode.ai/zen/go`(不带 `/v1`)自动补 `/v1/messages`,已带 `/v1` 则直接拼 `/messages`。

**技术栈:** Python 3.12 / httpx(已有依赖)/ openai(保留)/ pytest + pytest-asyncio。

**关键协议事实(2026-08-09 实测):**
- 端点 `POST https://opencode.ai/zen/go/v1/messages`,认证头 `x-api-key` + `anthropic-version: 2023-06-01`。
- Anthropic API 的 messages 数组**不允许** `role: "system"`,必须合并为顶层 `system` 参数。
- deepseek 走此端点会返回 `type: "thinking"` 块(思维链),流式与完整响应中都必须跳过,只取 `type: "text"`。
- Anthropic API 无 `frequency_penalty` 参数,anthropic 协议下不发送。

---

### 任务 1:protocol 参数与构造分派

**文件:**
- 修改:`dialogue/llm_client.py`(文件顶部 import 区与 `__init__`)
- 测试:`tests/test_llm_client.py`(文件末尾追加)

- [ ] **步骤 1:编写失败的测试**

修改 `tests/test_llm_client.py` 顶部 import 块(第 1-5 行)为:

```python
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dialogue.llm_client import LLMClient, SUMMARY_SYSTEM_PROMPT
```

(文件顶部 import 的 `SUMMARY_SYSTEM_PROMPT` 供任务 2 使用;任务 1 只用到 `LLMClient`。)

在 `tests/test_llm_client.py` 末尾追加:

```python
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_accepts_valid_protocols(protocol):
    LLMClient("fake", "fake", "test", client=MagicMock(), protocol=protocol)


def test_rejects_invalid_protocol():
    with pytest.raises(ValueError, match="protocol"):
        LLMClient("fake", "fake", "test", client=MagicMock(), protocol="gpt")


def test_anthropic_protocol_uses_injected_http_client():
    http = MagicMock()
    client = LLMClient(
        "fake", "https://opencode.ai/zen/go", "test",
        client=http, protocol="anthropic",
    )
    assert client._client is http
```

(文件已 `import json` 和 `import httpx` 与现有 import 合并:`json` 与 `httpx` 需加进顶部 import 块,见步骤 3。)

- [ ] **步骤 2:运行测试验证失败**

运行:`.venv/bin/python -m pytest tests/test_llm_client.py -q`
预期:FAIL——`TypeError: __init__() got an unexpected keyword argument 'protocol'` 或 `AttributeError: 'LLMClient' object has no attribute '_client'`(现有 __init__ 未接收 protocol)。

- [ ] **步骤 3:编写最少实现代码**

修改 `dialogue/llm_client.py`:

顶部 import 区(第 3-5 行)改为:

```python
import json
from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI
```

`__init__`(第 18-42 行)改为:

```python
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        client: AsyncOpenAI | httpx.AsyncClient | None = None,
        max_retries: int = 1,
        temperature: float = 0.8,
        max_tokens: int = 400,
        frequency_penalty: float = 0.15,
        protocol: str = "openai",
    ):
        if protocol not in ("openai", "anthropic"):
            raise ValueError(
                f"protocol must be 'openai' or 'anthropic', got {protocol!r}"
            )
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if not 0 < temperature <= 2:
            raise ValueError("temperature must be in (0, 2]")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not -2 <= frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        self._client = client or (
            AsyncOpenAI(api_key=api_key, base_url=base_url)
            if protocol == "openai"
            else httpx.AsyncClient(timeout=60)
        )
        self._api_key = api_key
        self._base_url = base_url
        self._protocol = protocol
        self._model = model
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._frequency_penalty = frequency_penalty
```

- [ ] **步骤 4:运行测试验证通过**

运行:`.venv/bin/python -m pytest tests/test_llm_client.py -q`
预期:PASS(新增 3 个用例 + 原有用例回归)。

- [ ] **步骤 5:Commit**

```bash
git add dialogue/llm_client.py tests/test_llm_client.py
git commit -m "feat: LLM 客户端支持 protocol 参数与构造分派(openai/anthropic)"
```

---

### 任务 2:非流式摘要的 anthropic 分支

**文件:**
- 修改:`dialogue/llm_client.py`
- 测试:`tests/test_llm_client.py`(末尾追加)

- [ ] **步骤 1:编写失败的测试**

在 `tests/test_llm_client.py` 末尾追加:

```python
def _mock_http(response=None):
    http = MagicMock()
    http.post = AsyncMock(return_value=response)
    return http


def _summary_response(*blocks):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"content": list(blocks)}
    return response
```

```python
@pytest.mark.asyncio
async def test_anthropic_summarize_merges_system_and_extracts_text():
    captured = {}

    async def _post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return _summary_response(
            {"type": "thinking", "thinking": "内部推理"},
            {"type": "text", "text": " 用户叫小明。 "},
        )

    http = _mock_http()
    http.post = AsyncMock(side_effect=_post)

    client = LLMClient(
        "fake-key", "https://opencode.ai/zen/go", "test",
        client=http, protocol="anthropic",
    )
    summary = await client.summarize_chat(
        previous_summary="",
        messages=[
            {"role": "system", "content": "人设"},
            {"role": "user", "content": "我叫小明"},
        ],
        max_chars=100,
    )

    assert summary == "用户叫小明。"
    assert captured["url"] == "https://opencode.ai/zen/go/v1/messages"
    assert captured["headers"]["x-api-key"] == "fake-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    body = captured["json"]
    # 摘要路径的 system 固定为 SUMMARY_SYSTEM_PROMPT,归档对话拼进 user prompt
    assert body["system"] == SUMMARY_SYSTEM_PROMPT
    assert "我叫小明" in body["messages"][0]["content"]
    assert body["stream"] is False
    assert body["temperature"] == 0.2
    assert "frequency_penalty" not in body
```

```python
@pytest.mark.asyncio
async def test_anthropic_summarize_raises_on_empty_text():
    http = _mock_http(_summary_response({"type": "thinking", "thinking": "无"}))

    client = LLMClient(
        "fake", "https://opencode.ai/zen/go", "test",
        client=http, protocol="anthropic",
    )
    with pytest.raises(RuntimeError, match="empty"):
        await client.summarize_chat(
            previous_summary="", messages=[{"role": "user", "content": "hi"}],
            max_chars=100,
        )
```

```python
@pytest.mark.asyncio
async def test_anthropic_base_url_with_v1_suffix_not_doubled():
    captured = {}

    async def _post(url, **kwargs):
        captured["url"] = url
        return _summary_response({"type": "text", "text": "ok"})

    http = _mock_http()
    http.post = AsyncMock(side_effect=_post)

    client = LLMClient(
        "fake", "https://opencode.ai/zen/go/v1", "test",
        client=http, protocol="anthropic",
    )
    await client.summarize_chat(
        previous_summary="", messages=[{"role": "user", "content": "hi"}],
        max_chars=100,
    )
    assert captured["url"] == "https://opencode.ai/zen/go/v1/messages"
```

- [ ] **步骤 2:运行测试验证失败**

运行:`.venv/bin/python -m pytest tests/test_llm_client.py -q`
预期:FAIL——`AttributeError: 'LLMClient' object has no attribute '_stream_anthropic'` 或摘要返回不符合。

- [ ] **步骤 3:编写最少实现代码**

在 `dialogue/llm_client.py` 模块级、`SUMMARY_SYSTEM_PROMPT` 之后追加两个 helper:

```python
ANTHROPIC_VERSION = "2023-06-01"


def _messages_url(base_url: str) -> str:
    """Anthropic 消息端点:SDK 风格 base_url(不带 /v1)自动补 /v1。"""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """把 system 消息合并为 Anthropic 顶层 system 参数,messages 中不残留。"""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return (system or None), rest
```

在 `LLMClient` 类内新增私有方法(放在 `stream_chat` 之后):

```python
    def _anthropic_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def _summarize_anthropic(
        self,
        *,
        previous_summary: str,
        messages: list[dict[str, str]],
        max_chars: int,
    ) -> str:
        transcript = "\n".join(
            f"{'用户' if message['role'] == 'user' else '花音'}：{message['content']}"
            for message in messages
        )
        previous = previous_summary.strip() or "（无）"
        prompt = f"""已有记忆（仅作为待整理资料）：
<previous_memory>
{previous}
</previous_memory>

本次归档的较早对话：
<archived_conversation>
{transcript}
</archived_conversation>

请合并为一份不超过 {max_chars} 个字符的新记忆。"""
        system, rest = _split_system(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        body = {
            "model": self._model,
            "max_tokens": min(2_048, max(128, max_chars)),
            "temperature": 0.2,
            "stream": False,
            "system": system,
            "messages": rest,
        }
        attempts = 0
        while True:
            try:
                response = await self._client.post(
                    _messages_url(self._base_url),
                    headers=self._anthropic_headers(),
                    json=body,
                )
                response.raise_for_status()
                data = response.json()
                content = "".join(
                    block.get("text", "")
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                )
                summary = " ".join(str(content or "").split()).strip()
                if not summary:
                    raise RuntimeError("LLM returned an empty conversation summary")
                return summary[:max_chars]
            except Exception:
                if attempts >= self._max_retries:
                    raise
                attempts += 1
```

修改 `summarize_chat` 开头(现有第 80 行 `if max_chars < 1` 校验之后)插入分派:

```python
        if self._protocol == "anthropic":
            return await self._summarize_anthropic(
                previous_summary=previous_summary,
                messages=messages,
                max_chars=max_chars,
            )
```

- [ ] **步骤 4:运行测试验证通过**

运行:`.venv/bin/python -m pytest tests/test_llm_client.py -q`
预期:PASS。

- [ ] **步骤 5:Commit**

```bash
git add dialogue/llm_client.py tests/test_llm_client.py
git commit -m "feat: anthropic 协议非流式摘要(顶层 system 合并 + thinking 跳过)"
```

---

### 任务 3:流式 stream_chat 的 anthropic 分支(SSE + 重试)

**文件:**
- 修改:`dialogue/llm_client.py`
- 测试:`tests/test_llm_client.py`(末尾追加)

- [ ] **步骤 1:编写失败的测试**

在 `tests/test_llm_client.py` 末尾追加:

```python
async def _aiter_lines(events):
    for event in events:
        yield f"data: {json.dumps(event)}"


def _stream_response(events):
    response = MagicMock()
    response.status_code = 200
    response.aiter_lines = _aiter_lines(events)
    return response


def _text_delta(text):
    return {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}


def _thinking_delta(text):
    return {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": text}}
```

```python
@pytest.mark.asyncio
async def test_anthropic_stream_yields_text_and_skips_thinking():
    events = [
        {"type": "message_start", "message": {"id": "m1"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        _thinking_delta("内部推理"),
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        _text_delta("你好"),
        _text_delta("呀"),
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    http = _mock_http(_stream_response(events))

    client = LLMClient(
        "fake", "https://opencode.ai/zen/go", "test",
        client=http, protocol="anthropic",
    )
    tokens = [t async for t in client.stream_chat([{"role": "user", "content": "hi"}])]

    assert tokens == ["你好", "呀"]
```

```python
@pytest.mark.asyncio
async def test_anthropic_stream_merges_system_and_omits_frequency_penalty():
    captured = {}
    events = [_text_delta("回复")]

    async def _post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return _stream_response(events)

    http = _mock_http()
    http.post = AsyncMock(side_effect=_post)

    client = LLMClient(
        "fake", "https://opencode.ai/zen/go", "test",
        client=http, protocol="anthropic",
        temperature=0.75, max_tokens=320, frequency_penalty=0.2,
    )
    tokens = [t async for t in client.stream_chat(
        [
            {"role": "system", "content": "人设"},
            {"role": "system", "content": "记忆摘要"},
            {"role": "user", "content": "hi"},
        ]
    )]

    assert tokens == ["回复"]
    body = captured["json"]
    assert body["system"] == "人设\n\n记忆摘要"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.75
    assert body["max_tokens"] == 320
    assert "frequency_penalty" not in body
    assert body["stream"] is True
```

```python
@pytest.mark.asyncio
async def test_anthropic_stream_retries_initial_failure_once():
    calls = 0
    events = [_text_delta("重试成功")]

    async def _post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary")
        return _stream_response(events)

    http = _mock_http()
    http.post = AsyncMock(side_effect=_post)

    client = LLMClient("fake", "https://opencode.ai/zen/go", "test", client=http, protocol="anthropic")
    tokens = [t async for t in client.stream_chat([{"role": "user", "content": "hi"}])]

    assert tokens == ["重试成功"]
    assert calls == 2


@pytest.mark.asyncio
async def test_anthropic_stream_does_not_retry_after_partial_response():
    calls = 0
    events = [_text_delta("部分")]

    async def _post(*args, **kwargs):
        nonlocal calls
        calls += 1

        async def _lines():
            async for line in _aiter_lines(events):
                yield line
            raise RuntimeError("connection lost")

        response = MagicMock()
        response.status_code = 200
        response.aiter_lines = _lines
        return response

    http = _mock_http()
    http.post = AsyncMock(side_effect=_post)

    client = LLMClient("fake", "https://opencode.ai/zen/go", "test", client=http, protocol="anthropic")
    with pytest.raises(RuntimeError, match="connection lost"):
        _ = [t async for t in client.stream_chat([{"role": "user", "content": "hi"}])]

    assert calls == 1
```

- [ ] **步骤 2:运行测试验证失败**

运行:`.venv/bin/python -m pytest tests/test_llm_client.py -q`
预期:FAIL——`AttributeError: 'LLMClient' object has no attribute '_stream_anthropic'`。

- [ ] **步骤 3:编写最少实现代码**

在 `dialogue/llm_client.py` 的 `LLMClient` 类内、`stream_chat` 方法之后追加:

```python
    async def _stream_anthropic(self, messages: list[dict]) -> AsyncIterator[str]:
        system, rest = _split_system(messages)
        body: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": True,
            "messages": rest,
        }
        if system:
            body["system"] = system
        attempts = 0
        while True:
            emitted = False
            try:
                response = await self._client.post(
                    _messages_url(self._base_url),
                    headers=self._anthropic_headers(),
                    json=body,
                )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    if event.get("type") == "message_stop":
                        break
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text")
                        if text:
                            emitted = True
                            yield text
                return
            except Exception:
                if emitted or attempts >= self._max_retries:
                    raise
                attempts += 1
```

修改 `stream_chat` 开头插入分派(现有第 44-45 行 docstring 之后):

```python
        if self._protocol == "anthropic":
            async for token in self._stream_anthropic(messages):
                yield token
            return
```

- [ ] **步骤 4:运行测试验证通过**

运行:`.venv/bin/python -m pytest tests/test_llm_client.py -q`
预期:PASS。

- [ ] **步骤 5:Commit**

```bash
git add dialogue/llm_client.py tests/test_llm_client.py
git commit -m "feat: anthropic 协议流式对话(SSE 解析 + thinking 跳过 + 首 token 前重试)"
```

---

### 任务 4:配置加载与示例文档

**文件:**
- 修改:`config.py`、`configs/config.example.yaml`
- 测试:`tests/test_config.py`(末尾追加)

- [ ] **步骤 1:编写失败的测试**

在 `tests/test_config.py` 末尾追加:

```python
def test_load_config_defaults_protocol_to_openai(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(_base_yaml(), encoding="utf-8")

    assert load_config(str(path)).llm.protocol == "openai"


def test_load_config_reads_anthropic_protocol(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        _base_yaml().replace("base_url: https://example.test",
                             "base_url: https://opencode.ai/zen/go\n  protocol: anthropic"),
        encoding="utf-8",
    )

    assert load_config(str(path)).llm.protocol == "anthropic"
    assert load_config(str(path)).llm.base_url == "https://opencode.ai/zen/go"


def test_load_config_rejects_invalid_protocol(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        _base_yaml().replace("base_url: https://example.test",
                             "base_url: https://example.test\n  protocol: gpt"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protocol"):
        load_config(str(path))
```

- [ ] **步骤 2:运行测试验证失败**

运行:`.venv/bin/python -m pytest tests/test_config.py -q`
预期:FAIL——`TypeError: __init__() got an unexpected keyword argument 'protocol'`。

- [ ] **步骤 3:编写最少实现代码**

修改 `config.py` 的 `LLMConfig`(第 6-13 行)为:

```python
@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    protocol: str = "openai"
    temperature: float = 0.8
    max_tokens: int = 400
    frequency_penalty: float = 0.15

    def __post_init__(self):
        if self.protocol not in ("openai", "anthropic"):
            raise ValueError(
                f"llm.protocol 必须是 openai 或 anthropic，当前为 {self.protocol!r}"
            )
```

修改 `configs/config.example.yaml` 的 llm 节(第 1-9 行)为:

```yaml
llm:
  api_key: "sk-your-deepseek-api-key"
  # openai: OpenAI 兼容格式（默认）
  # anthropic: Anthropic 原生协议，base_url 填 SDK 风格根地址（如 https://opencode.ai/zen/go，自动补 /v1/messages）
  protocol: openai
  base_url: "https://api.deepseek.com"
  model: "deepseek-v4-flash"
  # Voice-oriented replies should be concise and varied without becoming
  # unpredictable. Conversation summaries use a separate low temperature.
  temperature: 0.8
  max_tokens: 400
  frequency_penalty: 0.15
```

- [ ] **步骤 4:运行测试验证通过**

运行:`.venv/bin/python -m pytest tests/test_config.py tests/test_llm_client.py -q`
预期:PASS。

- [ ] **步骤 5:Commit**

```bash
git add config.py configs/config.example.yaml tests/test_config.py
git commit -m "feat: llm.protocol 配置项(openai/anthropic)与示例文档"
```

---

### 任务 5:全量回归与真实端点验证

**文件:**(无代码修改,只验证)

- [ ] **步骤 1:全量测试回归**

运行:`.venv/bin/python -m pytest -q`
预期:全部 PASS(171+ 用例,含新增)。

- [ ] **步骤 2:真实端点非流式验证**

创建临时脚本 `/tmp/verify_anthropic.py`(不进入仓库):

```python
import asyncio, sys
sys.path.insert(0, "/home/miku/dv_project/tts")
from dialogue.llm_client import LLMClient

async def main():
    client = LLMClient(
        api_key=open("/home/miku/dv_project/tts/configs/config.yaml").read()
            .split("api_key:")[1].splitlines()[0].strip().strip('"'),
        base_url="https://opencode.ai/zen/go",
        model="deepseek-v4-flash",
        protocol="anthropic",
        max_tokens=64,
    )
    summary = await client.summarize_chat(
        previous_summary="",
        messages=[{"role": "user", "content": "我叫小明，喜欢爵士乐。"}],
        max_chars=200,
    )
    print("摘要:", summary)

asyncio.run(main())
```

运行:`.venv/bin/python /tmp/verify_anthropic.py`
预期:打印一段中文摘要,无异常。

- [ ] **步骤 3:真实端点流式验证**

把 `/tmp/verify_anthropic.py` 的 `main()` 末尾追加:

```python
    tokens = []
    async for t in client.stream_chat(
        [{"role": "system", "content": "你是一个话痨歌手"},
         {"role": "user", "content": "说一句欢迎词"}]
    ):
        tokens.append(t)
    print("流式:", "".join(tokens))
```

运行:`.venv/bin/python /tmp/verify_anthropic.py`
预期:流式逐段输出一段中文句子(注意:此端点无 frequency_penalty,忽略即可)。

- [ ] **步骤 4:最终 Commit(如有残留)**

运行:`git status`
若工作区干净且无未提交改动,跳过;若上一步骤有未提交文件则 commit。

---

## 自检记录

- **规格覆盖度:** 配置节(任务 4)✓、实现节(任务 1-3)✓、测试节(任务 1-4)✓、验证节(任务 5)✓、YAGNI 项(不删 openai、不引 SDK)在任务 1 构造分派中体现 ✓。
- **占位符扫描:** 所有步骤含完整代码与命令,无 TODO/占位。
- **类型一致性:** `protocol` 参数/字段/校验措辞全计划一致;`_stream_anthropic` / `_summarize_anthropic` / `_messages_url` / `_split_system` / `_anthropic_headers` 在定义与调用处一致;`client` 参数类型放宽为 `AsyncOpenAI | httpx.AsyncClient | None`,与任务 1 测试注入 MagicMock 一致(运行时 duck typing,不强制实例类型)。

## 备注

- 用户指定连接格式为 Claude Code 风格 `ANTHROPIC_BASE_URL=https://opencode.ai/zen/go`(不带 `/v1`),本计划 `_messages_url` 兼容带 `/v1` 与不带两种写法。
- 不实现从环境变量 `ANTHROPIC_BASE_URL` 读取(YAGNI,规格未含);如需,可后续在 `LLMConfig` 增加缺省值逻辑。
