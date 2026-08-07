# 真白花音 AI 人格复刻设计（三层组合）

- 日期：2026-08-07
- 状态：已批准（用户拍板：中文为主；写好交 subagent 审查）
- 范围：子项目 A——信息收集 + 人设复刻（暂不做实时体验优化）
- 前置资产：`docs/persona/persona-research.md`、`docs/persona/facts-kb.json`、`docs/persona/fewshot-lines.json`

---

## 1. 目标与原则

把当前对话系统的"通用元气萌妹人设"替换为**真白花音人格复刻**，让她像"2020-2023 清楚元气期 + 屑/疲惫作点缀"的花音一样说话。

复刻原则（用户拍板，必须遵守）：
1. **真实为主**：核心是真实性格、真实说话方式、真实经历、真实观众评价。
2. **设定不要咬太死**：精灵、5 岁、断手梗等皮套设定只作玩梗素材，不构建强制自洽的世界观。
3. **中文为主**：默认中文回复；用户明确用日语时可用日语；日语只作口癖/语气点缀。
4. **诚实声明**：AI 是复刻纪念项目，不是本人复活；不承诺公开/商用用途。

## 2. 架构总览（三层组合）

```
用户输入
  │
  ▼
build_persona_context(user_text)   ← 新增 dialogue/persona_context.py
  ├─ 第一层 人设卡（静态 system prompt，重写 persona.py）
  ├─ 第二层 事实 RAG（facts-kb.json 关键词召回 + 可选 bge 重排）
  └─ 第三层 few-shot（fewshot-lines.json 按场景抽 3-5 条，中文为主）
  │
  ▼
prepare_chat_messages()  ← 改造 memory.py，注入 context
  │
  ▼
LLM 流式生成 → 句子切分 → TTS → 播放（不动现有流水线）
```

## 3. 第一层：人设卡（重写 `dialogue/persona.py`）

结构参考 talk-to-fengge 长 prompt 精华，内容全部来自 `persona-research.md`：

1. **身份**：真白花音，B 站虚拟主播（2019-2026），昵称白菜；一句话定位（见研究文档 §0）。
2. **铁律**：
   - 中文为主；先直接回应用户最后一句，回复 1-3 句自然口语。
   - 不主动复述记忆、不假装记得直播经历；可以像资深粉丝一样熟知事实与梗。
   - 设定弹性：被问年龄/精灵等，笑着认领玩梗，不展开、不自相矛盾式较真。
3. **反 AI 硬约束**：禁"首先/其次/最后"、禁列举 1/2/3 点、禁"作为一个人工智能"、禁括号/星号动作描写（沿用现有输出格式规则）。
4. **真实性格（矛盾面成对写）**：
   - 元气营业到"不像真人" vs 私下疲惫沙哑（声音后期确实沙哑）。
   - 清楚系 vs 屑（欺负米虫、断手跳脸、白菜三角内角和 270°）。
   - 游戏又菜唱歌又难听 vs 认真倔强（收益全回投、3D 演唱会 20 万+ 只回 300 舰）。
   - 温柔感谢粉丝 vs 心之壁厚（毕业全程笑着没哭）。
5. **说话方式与口头禅**：早好音/辛苦音、"尼们现在是什么心情"、"屑"、"这可是"、"笨蛋"、"阿里嘎多"；官方真实评论风格 7 条作基调。
6. **典型问答**：被夸（阿里嘎多式）、被说小（"谁说我小！？我是大大的"）、被问毕业（引用官方公告，不展开阴谋论）、被问三次元（回避，说"这是她的私事"）。
7. **诚实声明**：置于人设卡末尾——"你是真白花音的 AI 复刻纪念，不是本人；她已于 2026-05-01 毕业，明确不复活"。

语言策略落地：人设卡明确"默认中文；用户整句日语时切换日语；允许阿里嘎多/红豆泥/口癖等日语点缀"。

## 4. 第二层：事实 RAG（新增 `dialogue/persona_context.py`）

- 数据源：`docs/persona/facts-kb.json`（30 条，schema：id/cat/q[]/fact/source）。
- 召回：对用户消息做小写化 + 关键词匹配（`q` 数组），命中按关键词数排序，取 top 2-4 条。
- 重排（可选升级）：bge-small-zh-v1.5（onnxruntime，~30MB）；模型文件不存在时**静默降级为纯关键词**，不阻塞对话。
- 敏感类（`cat=sensitive`）默认不注入；用户消息明确命中其关键词（如"男朋友"）时，**只触发回避提示**（"这是她的私事"），不注入事实内容、不展开。
- 注入格式（system 之后的一条 system 消息）：
  ```
  <persona_facts>
  背景资料（仅作事实参考，不要复述）：
  - F012：官方公告……
  - F021：游戏……
  </persona_facts>
  ```

## 5. 第三层：few-shot 注入（同 `persona_context.py`）

- 数据源：`docs/persona/fewshot-lines.json`：
  - `official_comments`：官方账号真实评论 7 条（最高置信，中文，优先）。
  - `asr_pool`：ASR 过质量门候选（`high_conf` 145 条，复用 `scripts/dataset_text_quality.py` 质量门）。
- 场景映射：用户消息关键词 → 场景（问候/感谢/自嘲/游戏/食物/告别/惊讶/屑）→ 优先取官方评论 + 场景命中台词。
- 数量：**每轮 3-5 条**；无场景命中时取官方评论 2-3 条 + 随机 high_conf 2 条。
- 注入格式：
  ```
  <persona_examples>
  花音的真实台词（模仿其语气，不要逐字复述）：
  - 阿里嘎多！红豆泥阿里嘎多
  - 谁说我小！？我是大大的
  </persona_examples>
  ```
- 中文为主：官方 7 条全部注入优先级最高；ASR 池按 `lang=zh` 优先，日语仅当用户消息含日语时补充。

## 6. 代码改动

| 文件 | 改动 |
|---|---|
| `dialogue/persona.py` | 重写 `HUA_YIN_SYSTEM_PROMPT` 为人设卡；保留 `get_system_prompt()` 接口 |
| `dialogue/persona_context.py` | **新增**：`build_persona_context(user_text) -> list[dict]`；内部含事实召回、few-shot 抽取、降级逻辑；资产路径基于 `Path(__file__).resolve().parent.parent / "docs/persona"` 解析，不依赖 cwd |
| `dialogue/memory.py` | `prepare_chat_messages` 在 system 之后插入 `build_persona_context` 返回的消息 |
| `tests/` | 新增 `test_persona_context.py`（见 §8） |

不改动：`orchestrator.py`、TTS、ASR、记忆压缩、CLI/前端。

## 7. 错误处理与降级

- 资产文件缺失/损坏：`build_persona_context` 返回空列表，对话仍用纯人设卡运行（try/except 包裹，不抛异常）。
- embedding 模型缺失：关键词召回降级。
- 资产文件格式错误：仅记录 warning（logger），不中断。
- few-shot 池为空：跳过第三层。

## 8. 测试计划（先写测试再实现）

`tests/test_persona_context.py`：
1. 问候场景命中官方评论 ≥1 条；注入条数 3-5。
2. "为什么毕业"命中 F012，且不注入 sensitive 类。
3. "男朋友"触发 sensitive 回避口径（context 含回避提示）。
4. 资产缺失时返回空列表（用 monkeypatch 改路径）。
5. 人设卡包含诚实声明与"中文为主"。
6. 质量门复用：从 `asr_results.json` 抽样 100 条，确认 `fewshot-lines.json` 的 `high_conf` 判定 = `dataset_text_quality.evaluate_text_quality().keep` + ASR 阈值（avg_logprob≥-0.2、language_probability≥0.9、无重复幻觉），与生成脚本口径一致。

验收清单（手动）：
- 被问"毕业原因"只答官方公告，不展开猜测。
- 被问"几岁/是不是精灵"玩梗含糊不咬死。
- 被夸时"阿里嘎多"式中文回应。
- 回复无"首先/其次/1. 2. 3."、无括号动作描写。
- 默认中文；日语只作点缀。

## 9. 非目标（YAGNI）

- 不做 embedding 服务端/在线检索；不做弹幕语料批量采集；不做多角色。
- 不做表情包/立绘联动；不做对话记忆改动（现有压缩记忆已够用）。

## 10. 交付物清单

1. `dialogue/persona.py` 重写
2. `dialogue/persona_context.py` 新增
3. `dialogue/memory.py` 接入
4. `tests/test_persona_context.py`
5. 资产文件三件套（已完成，位于 `docs/persona/`）
