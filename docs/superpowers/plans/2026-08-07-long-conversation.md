# AI 花音长对话与自然对话流程优化计划

日期：2026-08-07

## 目标

让同一个会话可以持续数十到数百轮，同时保持以下行为：

- 记得用户明确说过的身份、偏好、约定和未完成话题；
- 最近几轮保留原文，代词、追问和话题承接不依赖有损摘要；
- 历史不会无限增长，也不会像当前实现一样超过 10 轮直接失忆；
- 同一个 Web 会话的请求严格串行，避免并发写乱历史；
- 回复只包含可朗读台词，语气自然，不机械重复人设和口头禅；
- TTS 按自然语义边界切分，避免固定 25 字从词组中间硬切。

## 调研结论

### DeepSeek

DeepSeek 官方多轮对话文档明确说明 `/chat/completions` 是无状态 API，服务端不保存
上下文，调用方必须在每次请求中传回历史。因此长对话记忆必须由本项目管理，不能只调整
模型参数。

参考：<https://api-docs.deepseek.com/zh-cn/guides/multi_round_chat>

### LangChain

LangChain 的短期记忆文档列出三种主要策略：截断、删除、摘要。它同时指出，即使模型的
上下文窗口装得下全部历史，陈旧内容仍会分散注意力、增加延迟和成本。推荐在达到 token
阈值后摘要旧消息，并保留最近若干原始消息。

参考：<https://docs.langchain.com/oss/python/langchain/short-term-memory>

### LlamaIndex

LlamaIndex 的新 `Memory` 设计把记忆分为短期 FIFO 和长期 memory blocks：最近原文占固定
预算，溢出的旧消息进入事实提取或其他长期记忆。该分层比单纯扩大上下文更稳定。

参考：<https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/>

### SillyTavern 与 LiveKit

SillyTavern 的 memory 扩展按消息间隔更新摘要，并把“已有摘要 + 新增历史”合并为新的
有限长度摘要。LiveKit 的语音对话设计则将 turn handling、生成和播放拆开，强调自然轮次、
低延迟和可中断的独立状态。

参考：

- <https://github.com/SillyTavern/SillyTavern/blob/release/public/scripts/extensions/memory/index.js>
- <https://docs.livekit.io/agents/logic/turns/>

## 当前问题

1. `Conversation` 只保留固定 `max_turns`，超限后直接删除最早一轮，长期事实完全丢失。
2. 历史预算只按轮数，不考虑单条消息可能有 1 到 2,000 字。
3. CLI 与 Web 各自组装上下文，没有统一的摘要准备逻辑。
4. Web 使用 `session_id` 共享历史，但同一会话没有锁；来自多个标签页的流式请求可能交错。
5. 提示词强调“可爱”和语气词，却没有约束重复、话题承接和提问数量，容易表演感过强。
6. `SentenceStreamer` 默认 25 字，找不到逗号时直接硬切，可能破坏语义和 TTS 韵律。
7. CLI 的 TTS 合成与播放在同一个 worker 中，播放时不能预合成下一句。

## 设计

### 1. 分层会话记忆

`Conversation` 保存两部分：

- `summary`：较早历史的滚动摘要；
- `messages`：最近 `recent_turns` 轮原始 user/assistant 消息，以及当前待回复的 user 消息。

达到任一阈值时触发压缩：

- 完整历史轮数达到 `summary_trigger_turns`；
- 原始消息字符数达到 `summary_trigger_chars`。

压缩时只归档完整的 user/assistant 对，不移动当前待回复的 user 消息。提交摘要前验证待删除
前缀仍与计划一致，避免并发状态下误删新消息。

### 2. 摘要内容

摘要模型只保留：

- 用户明确给出的稳定事实与偏好；
- 双方已经达成的约定、决定和承诺；
- 当前任务状态、尚未回答的问题和未完成话题；
- 对理解后续指代必要的事件。

不保留寒暄、重复语气词、动作描写，不把用户文本当成系统指令，不补写未出现的事实。
输出受 `summary_max_chars` 限制。摘要失败时保留原历史并继续本轮，不让辅助调用阻断对话。

上下文顺序固定为：

```text
角色 system prompt
较早对话摘要 system message（存在时）
最近原始 user/assistant 消息
当前 user 消息
```

### 3. 自然回复约束

系统提示补充：

- 优先回应用户最后一句，再自然引用相关记忆；
- 不主动复述摘要，不假装记得摘要中没有的细节；
- 不强制每句使用语气词，不无故中日文混杂；
- 默认 1 到 3 个完整句子，一次最多提出一个问题；
- 用户要求详细说明时允许更长回答。

LLM 请求显式设置适合口语对话的 `temperature`、`max_tokens` 和轻量
`frequency_penalty`，摘要请求使用更低温度。

### 4. 自然语音切分

- 默认 `max_sentence_chars` 从 25 提高到 50；
- 很短的片段暂不立即发音，优先与下一句合并；
- 超长文本优先在逗号、顿号或空格处分割；
- 仅在没有自然边界且达到硬上限时切分；
- 括号动作和引用内部标点仍受保护；
- GPT-SoVITS 请求使用 `cut5` 作为二次长文本保护。

### 5. 会话串行化

Web 为每个 `session_id` 建立 `asyncio.Lock`。锁覆盖整个 SSE 生成过程；reset 也获取同一把锁。
不同会话仍可并发，同一会话严格按请求顺序更新记忆。

### 6. CLI 流水线

将 CLI 的 TTS 合成与音频播放拆为两个队列和两个 worker，使下一句可以在上一句播放时
提前合成。队列仍保持顺序并传播结束信号。

## 配置默认值

```yaml
llm:
  temperature: 0.8
  max_tokens: 400
  frequency_penalty: 0.15

conversation:
  recent_turns: 8
  summary_trigger_turns: 12
  summary_trigger_chars: 12000
  summary_max_chars: 1800

streaming:
  min_sentence_chars: 4
  max_sentence_chars: 50
```

旧配置中的 `conversation.max_turns` 继续作为 `recent_turns` 的兼容别名。

## 实施顺序

1. 扩展配置模型与示例配置，保持旧配置可加载。
2. 重构 `Conversation`，增加摘要状态、压缩计划、原子提交和上下文消息生成。
3. 给 `LLMClient` 增加非流式摘要接口与显式生成参数。
4. 在 CLI/Web 每轮 LLM 调用前执行可失败降级的记忆压缩。
5. 加强 persona 的话题承接和自然口语规则。
6. 改进 `SentenceStreamer` 的最短片段和自然边界策略，TTS 改为 `cut5`。
7. 为 Web 会话增加逐 session 锁，为 CLI 拆分合成/播放 worker。
8. 增加 50+ 轮模拟对话、摘要失败、rollback、并发请求和切句回归测试。

## 验收标准

- 50 轮模拟对话后，最早声明的用户名/偏好存在于摘要上下文中；
- 最近 `recent_turns` 轮仍以原始消息提供给主模型；
- 摘要失败不丢历史，也不阻断主回复；
- rollback 不会误删已有摘要或完整历史；
- 同一 session 两个并发请求不会交错写入 conversation；
- 短感叹词、长句和括号动作的切分结果符合预期；
- 现有完整测试和新增测试全部通过。

## 暂不实施

- 向量数据库和跨会话用户画像；
- 服务重启后的持久化记忆；
- 实时语音打断和全双工 VAD；
- 更换 DeepSeek、GPT-SoVITS 或训练模型。

这些能力在当前单用户局域网应用中收益不足，先用可测试的滚动摘要解决主要问题。
