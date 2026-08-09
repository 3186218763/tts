# 设计规格:LLM 客户端支持 Anthropic 原生协议

日期:2026-08-09
状态:已批准

## 目标

为对话核心的 LLM 客户端新增 **Anthropic 原生协议**(`/v1/messages`)支持,作为可配置选项,
通过 OpenCode Go 网关(`https://opencode.ai/zen/go/v1`)连接,模型保持 `deepseek-v4-flash`。
**不替换**现有 OpenAI 兼容实现,协议通过配置切换,上游调用方零改动。

## 背景验证(2026-08-09 实测)

- OpenCode Go OpenAI 兼容端点 `https://opencode.ai/zen/go/v1`(SDK 内部拼 `/chat/completions`),
  当前配置即此格式,`GET /v1/models` 返回 200、25 个模型。
- Anthropic 原生端点 `POST https://opencode.ai/zen/go/v1/messages`(`x-api-key` +
  `anthropic-version: 2023-06-01` 头)对 `deepseek-v4-flash` 实测 200,可用。
- **关键观察**:deepseek 走 Anthropic 协议时响应含 `type: "thinking"` 块(思维链),
  流式与完整响应中都必须跳过,只取 `type: "text"` 内容。

## 配置

```yaml
llm:
  protocol: anthropic        # 新增:openai(默认,兼容旧配置)| anthropic
  api_key: "sk-..."
  base_url: "https://opencode.ai/zen/go/v1"   # Anthropic 端点
  model: "deepseek-v4-flash"
  temperature: 0.8
  max_tokens: 400
  frequency_penalty: 0.15    # 仅 openai 协议生效;anthropic 协议忽略
```

- `LLMConfig` 新增 `protocol: str = "openai"` 字段,校验值 ∈ {openai, anthropic},非法值报错。
- 旧配置无 `protocol` 字段 → 默认 `openai`,行为完全不变。

## 实现

### `dialogue/llm_client.py`

公开接口不变,仅构造时按协议分派:

- `LLMClient.__init__(..., protocol: str = "openai")`:
  - `openai` → 现有 `AsyncOpenAI` 客户端(原逻辑不动)。
  - `anthropic` → 内部持有 `httpx.AsyncClient`,**不引入 anthropic SDK,零新增依赖**。
- `stream_chat(messages)`(签名不变):
  - 把消息列表中所有 `role == "system"` 的 content 拼接为顶层 `system` 参数(人设 +
    记忆摘要 + persona 上下文均在此列,自动合并;无 system 消息则省略)。
  - `POST {base_url.rstrip('/')}/messages`,headers:`x-api-key` / `anthropic-version:
    2023-06-01` / `content-type: application/json`。
  - body:`model` / `max_tokens` / `temperature` / `stream: true` / `messages`(过滤掉
    system 后) / `system`(如有)。
  - 逐行解析 SSE 事件:`content_block_delta` 事件中 `delta.type == "text"` 时产出
    `delta.text`;**`delta.type == "thinking"` 跳过**;`message_stop` 结束。
  - 重试语义不变:首 token 前失败可重试(`max_retries`),已产出则不重试。
  - `frequency_penalty` 在 anthropic 协议下忽略(不发送该字段)。
- `summarize_chat(...)`(签名不变):
  - 同协议发送非流式请求(`stream: false`),从响应 `content` 数组取 `type == "text"`
    的块拼接为摘要;`thinking` 块跳过。
  - 摘要为空 → 保持现有 `RuntimeError` 语义;`max_chars` 截断逻辑不变。

### `config.py`

- `LLMConfig` 加 `protocol: str = "openai"`,加载时校验合法值。

### `configs/config.example.yaml`

- `llm` 节加 `protocol` 字段与两种协议 base_url 示例注释。

## 测试

- `tests/test_llm_client.py` 新增 anthropic 协议用例(mock httpx 层,不发真实请求):
  1. 流式 text 增量正常产出;thinking 增量被跳过。
  2. 多个 system 消息合并为顶层 `system` 参数,messages 中不残留 system role。
  3. 非流式摘要:content 中混合 thinking + text 时只取 text;空摘要抛 RuntimeError。
  4. 重试:首 token 前失败重试至 `max_retries`,已产出后失败直接抛。
  5. 非法 `protocol` 值在构造/配置加载时报错;`openai` 路径行为不变(现有用例回归)。
- 原有 171 个测试全部保留,全量 `pytest -q` 跑绿。

## 验证

1. `pytest -q` 全绿(新旧用例)。
2. 用 `configs/config.yaml` 真实 key,以 `protocol: anthropic` 发起一次最小非流式
   调用,确认 200 与 text 内容;一次流式调用确认逐段输出。
3. Web 服务 `/healthz` 仍正常(不影响配置加载)。

## 不做的事(YAGNI)

- 不删除 OpenAI 兼容路径,不做协议自动探测/回退。
- 不引入 anthropic 官方 SDK。
- 不支持 Anthropic 的 tools/缓存控制等特性,按当前接口最小实现。
- 不改动 orchestrator / memory / conversation 等上游模块。
