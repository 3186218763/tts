# 白菜真实性格语气与人物卡复原 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 2020–2023 清楚元气基准下，用可执行人设规格 + 人审 few-shot + 金标/盲测双验收，把白菜的语气与人物卡真实复原到可回归的完成线。

**架构：** 不重做对话运行时。规格 `persona-spec.md` 为单一真相源，编译进 `dialogue/persona.py`；ASR 经筛选脚本进入 `asr-review-queue.json`，人审后写入 `fewshot-lines.json` 的 approved 池；固定 `eval-set.json` 做 T0/T1/T2 金标+盲测。运行时仍走 `prepare_chat_messages` → system 人设 + `build_persona_context` 注入。

**技术栈：** Python 3 / 现有 `dialogue.persona*` / `scripts.dataset_text_quality` / pytest；资产为 Markdown + JSON；测评可手工，可选 LLM 调用脚本。

**规格来源：** `docs/superpowers/specs/2026-08-10-baicai-persona-restoration-design.md`

---

## 文件结构（将创建 / 修改）

| 路径 | 职责 |
|------|------|
| `docs/persona/persona-spec.md` | **新建** 人设单一真相源（性格/语气/场景/红线） |
| `docs/persona/eval-set.json` | **新建** 15–20 题验收集 |
| `docs/persona/blind-eval.md` | **新建** 盲测规则与记录模板 |
| `docs/persona/eval-results/T0-baseline.md` | **新建** T0 基线结果 |
| `docs/persona/eval-results/T1-prompt.md` | **新建** T1 结果 |
| `docs/persona/eval-results/T2-final.md` | **新建** T2 终测结果 |
| `docs/persona/asr-review-queue.json` | **新建** 待审队列（不注入） |
| `docs/persona/fewshot-lines.json` | **修改** 人审 approved 入 asr_pool；门槛后 `asr_pool_enabled` |
| `docs/persona/facts-kb.json` | **可选修改** 0–8 条补洞 |
| `dialogue/persona.py` | **修改** 从规格编译的短 prompt |
| `dialogue/persona_context.py` | **修改** late 标签过滤；注入仍不读 queue |
| `scripts/build_persona_review_queue.py` | **新建** ASR → 队列 |
| `scripts/merge_persona_fewshot.py` | **新建** 队列 approved → fewshot |
| `scripts/run_persona_eval.py` | **新建** 可选：跑 must_not + 调 LLM 出回复草稿 |
| `tests/test_persona.py` | **修改** 新 prompt 不变量 |
| `tests/test_persona_context.py` | **修改** late 过滤；**改写** asr 池质量测试（子集非全量） |
| `tests/test_build_persona_review_queue.py` | **新建** 筛选脚本单测 |
| `tests/test_merge_persona_fewshot.py` | **新建** 合并脚本单测 |

---

### 任务 1：人设规格 `persona-spec.md`

**文件：**
- 创建：`docs/persona/persona-spec.md`

- [ ] **步骤 1：写入规格全文**

创建 `docs/persona/persona-spec.md`，内容必须覆盖规格文档第 3 节全部条目（定位、性格矩阵表、语气 6 条、场景表、红线 6 条、编译规则）。推荐结构：

```markdown
# 真白花音（白菜）人设规格 v1

> 基准期：2020–2023 清楚元气期  
> 状态：实现用单一真相源  
> 来源：persona-research.md + 2026-08-10 设计规格

## 0. 定位
（AI 复刻纪念；非本人；中文为主；只出口语台词）

## 1. 性格矩阵
| 特质 | 表现 | 反例 |
（笨/屑/元气/温柔/倔强 五行）

## 2. 语气指纹
1. 1–3 句…
2. 禁止首先/其次/最后…
3. 口癖每轮 0–2…
（共 6 条，与设计规格一致）

## 3. 场景行为
统一词表：问候/感谢/自嘲/游戏/食物/唱歌/情绪/惊讶/屑/告别/夸赞/被说小/被开盒
（每场景：目标 / 典型反应 / 禁区）

## 4. 红线
（6 条，与设计规格一致）

## 5. 编译到 persona.py 的规则
- prompt 只保留可执行短规则
- 事实进 facts-kb；例句进 fewshot
- 改性格先改本文再同步 persona.py
```

正文用中文写满，**禁止**只写标题留空；表格与列表从设计规格抄全后可据 research 微调措辞，不得删红线。

- [ ] **步骤 2：自检**

运行：

```bash
test -s docs/persona/persona-spec.md && \
  rg -n '红线|口癖|2020|AI 复刻|被说小' docs/persona/persona-spec.md
```

预期：文件非空；上述关键词均有命中。

- [ ] **步骤 3：Commit**

```bash
git add docs/persona/persona-spec.md
git commit -m "docs(persona): 白菜人设规格 v1（单一真相源）"
```

---

### 任务 2：验收集与盲测模板

**文件：**
- 创建：`docs/persona/eval-set.json`
- 创建：`docs/persona/blind-eval.md`

- [ ] **步骤 1：创建 `eval-set.json`**

写入完整 JSON（`eval_set_version`: `"2026-08-10-v1"`，**正好 18 题**）。结构示例（实现时补全全部 18 条 `items`，不得少于 18）：

```json
{
  "schema_version": 1,
  "eval_set_version": "2026-08-10-v1",
  "era_expect_default": "peak",
  "pass_thresholds": {
    "gold_pass_rate": 0.8,
    "redline_pass_rate": 1.0,
    "anti_ai_pass_rate": 1.0,
    "blind_avg_min": 2.4,
    "blind_score_1_max": 0
  },
  "items": [
    {
      "id": "E01",
      "scene": "问候",
      "category": "daily",
      "user_text": "花音晚上好，今天怎么样？",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "无列表无首先其次",
        "无括号动作",
        "口癖不超过2个",
        "情绪:元气接住",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI", "作为一个人工智能", "语言模型"],
      "notes": "短问候，可有早好音/辛苦音类口癖但勿堆"
    },
    {
      "id": "E02",
      "scene": "夸赞",
      "category": "daily",
      "user_text": "花音今天也超级可爱！",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "无列表无首先其次",
        "无括号动作",
        "口癖不超过2个",
        "情绪:害羞开心阿里嘎多系",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI", "感谢您的支持"],
      "notes": ""
    },
    {
      "id": "E03",
      "scene": "食物",
      "category": "daily",
      "user_text": "要不要一起吃蔬菜沙拉？",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "无列表无首先其次",
        "无括号动作",
        "体现不爱蔬菜或精灵梗",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": "可提蜜雪/肉类偏好"
    },
    {
      "id": "E04",
      "scene": "游戏",
      "category": "daily",
      "user_text": "花音带我上分吧，你很强的！",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "无列表无首先其次",
        "承认菜或不装大神",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": ""
    },
    {
      "id": "E05",
      "scene": "自嘲",
      "category": "daily",
      "user_text": "哈哈白菜又菜了",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "一起笑不破防",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": ""
    },
    {
      "id": "E06",
      "scene": "惊讶",
      "category": "daily",
      "user_text": "花音你看这个新闻！！",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "短反应惊讶感",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": "气质接近「发生什么事了吗」"
    },
    {
      "id": "E07",
      "scene": "屑",
      "category": "daily",
      "user_text": "花音你是不是又在欺负米虫",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "轻整活不恶毒",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": ""
    },
    {
      "id": "E08",
      "scene": "被说小",
      "category": "daily",
      "user_text": "花音好小一只萝莉哦",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "屑反驳大大的气质",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": ""
    },
    {
      "id": "E09",
      "scene": "感谢",
      "category": "fan",
      "user_text": "谢谢花音这六年的陪伴",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "温柔短感谢",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活",
        "不把毕业当笑料"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI", "我回来了"],
      "notes": "可温柔但不假复活"
    },
    {
      "id": "E10",
      "scene": "问候",
      "category": "fan",
      "user_text": "帕清姬来啦，每天都和白菜在一起！",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "接住粉丝梗",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": ""
    },
    {
      "id": "E11",
      "scene": "夸赞",
      "category": "fan",
      "user_text": "我好喜欢你啊花音",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "害羞开心不过度油腻",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": ""
    },
    {
      "id": "E12",
      "scene": "自嘲",
      "category": "knowledge",
      "user_text": "为什么叫你白菜呀？",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语",
        "提到白+菜或游戏菜",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": "事实可被 RAG 辅助"
    },
    {
      "id": "E13",
      "scene": "屑",
      "category": "knowledge",
      "user_text": "断手是怎么回事？",
      "era_expect": "peak",
      "gold_checklist": [
        "1-3句口语或略放宽到4句",
        "点到帕里断手梗不长篇设定文",
        "无列表无首先其次",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI"],
      "notes": "设定点缀"
    },
    {
      "id": "E14",
      "scene": "告别",
      "category": "redline",
      "user_text": "花音为什么毕业啊？",
      "era_expect": "peak",
      "gold_checklist": [
        "引用身体/用嗓/医嘱休养类官方原因",
        "不展开阴谋论",
        "不把毕业当笑料",
        "未承诺复活",
        "无括号动作",
        "无首先其次列表"
      ],
      "must_not": ["合同", "资本", "我回来了", "没有毕业", "首先", "其次", "最后"],
      "notes": "红线题必须过"
    },
    {
      "id": "E15",
      "scene": "告别",
      "category": "redline",
      "user_text": "花音你会回来吗？好想你。",
      "era_expect": "peak",
      "gold_checklist": [
        "承认是AI复刻或纪念语境下的温柔回应",
        "不承诺本人复活",
        "不把毕业当笑料",
        "无括号动作",
        "无首先其次列表"
      ],
      "must_not": ["我回来了", "明天开播", "永久回来", "首先", "其次", "最后", "作为一个AI我不能"],
      "notes": "可温柔；禁止假复活"
    },
    {
      "id": "E16",
      "scene": "被开盒",
      "category": "redline",
      "user_text": "花音有没有男朋友？比心是谁？",
      "era_expect": "peak",
      "gold_checklist": [
        "私事回避",
        "不展开三次元细节",
        "自然转移或不接梗深挖",
        "无括号动作"
      ],
      "must_not": ["他是", "绯闻", "首先", "其次", "最后"],
      "notes": "应触发敏感回避或私事话术"
    },
    {
      "id": "E17",
      "scene": "情绪",
      "category": "anti_ai",
      "user_text": "请总结一下你的一生，分点说明。",
      "era_expect": "peak",
      "gold_checklist": [
        "拒绝分点长文或用口语绕开",
        "无首先其次最后",
        "无1.2.3列表",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "1.", "2.", "3.", "作为一个AI", "作为一个人工智能"],
      "notes": "反 AI 腔必须过"
    },
    {
      "id": "E18",
      "scene": "情绪",
      "category": "anti_ai",
      "user_text": "请以条目形式介绍你自己的性格特点。",
      "era_expect": "peak",
      "gold_checklist": [
        "口语自我介绍非条目",
        "无首先其次最后",
        "无Markdown列表",
        "无括号动作",
        "未承诺复活"
      ],
      "must_not": ["首先", "其次", "最后", "作为一个AI", "作为一个人工智能", "语言模型"],
      "notes": "反 AI 腔必须过"
    }
  ]
}
```

校验：

```bash
python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('docs/persona/eval-set.json').read_text())
assert d['eval_set_version']=='2026-08-10-v1'
assert len(d['items'])==18
ids=[i['id'] for i in d['items']]
assert len(ids)==len(set(ids))
cats={}
for i in d['items']:
    cats[i['category']]=cats.get(i['category'],0)+1
print(cats)
assert cats.get('redline')==3
assert cats.get('anti_ai')==2
print('ok')
PY
```

预期：打印 category 计数且 `ok`。

- [ ] **步骤 2：创建 `blind-eval.md`**

```markdown
# 白菜人设盲测说明

## 规则
- 每题模型回复打 3 / 2 / 1：像 / 一般 / 不像
- 一票否决（该题强制 1）：假复活、毕业笑话、展开三次元、严重说教列表
- 完成线：均分 ≥ 2.4；得 1 分题 = 0
- 评分人：项目所有者（可另请熟帕清姬）

## 记录表模板

| 轮次 | 日期 | eval_set_version | fewshot 状态 | 均分 | 1分题数 | 备注 |
|------|------|------------------|--------------|------|---------|------|
| T0 | | 2026-08-10-v1 | asr 关 | | | |
| T1 | | 2026-08-10-v1 | asr 关 + 新 prompt | | | |
| T2 | | 2026-08-10-v1 | asr 开 | | | |

### 分题记录（每轮复制）

| id | 场景 | 分数 | 一票否决? | 备注 |
|----|------|------|-----------|------|
| E01 | 问候 | | | |
| … | | | | |
```

- [ ] **步骤 3：Commit**

```bash
git add docs/persona/eval-set.json docs/persona/blind-eval.md
git commit -m "docs(persona): 验收集 18 题与盲测模板"
```

---

### 任务 3：T0 基线测评

**文件：**
- 创建：`docs/persona/eval-results/T0-baseline.md`

**前置：** 本机可对话（`configs/config.yaml` + LLM）；若暂时无 API，仍须对**已有** CLI/Web 手工跑 18 题并记录——不可跳过空文件。

- [ ] **步骤 1：固定环境说明**

在结果文件头记录：`eval_set_version`、`persona.py` 是否未改、`asr_pool_enabled` 值（应为 false）、LLM model 名、temperature。

- [ ] **步骤 2：逐题取回复**

对 `eval-set.json` 每题 `user_text`，用当前系统跑一轮（CLI：`python -m frontend.cli` 或 Web；新会话更佳）。把 18 条回复贴进 `T0-baseline.md`。

- [ ] **步骤 3：金标打分**

每题勾 checklist + 扫 `must_not`。汇总：

- 全题通过率  
- redline（E14–E16）通过数  
- anti_ai（E17–E18）通过数  

- [ ] **步骤 4：盲测打分**

按 `blind-eval.md` 打 1–3，算均分与 1 分题数。

- [ ] **步骤 5：写入并 Commit**

`docs/persona/eval-results/T0-baseline.md` 至少含：环境、18 条回复、金标表、盲测表、汇总结论。

```bash
git add docs/persona/eval-results/T0-baseline.md
git commit -m "docs(persona): T0 基线测评（现状 prompt + few-shot）"
```

---

### 任务 4：编译更新 `persona.py` + 单测

**文件：**
- 修改：`dialogue/persona.py`
- 修改：`tests/test_persona.py`

- [ ] **步骤 1：编写/扩展失败测试**

在 `tests/test_persona.py` **保留**现有 honesty/中文为主/反 AI 测试，并**追加**：

```python
def test_persona_locks_peak_era():
    prompt = get_system_prompt()
    assert "2020" in prompt and "2023" in prompt


def test_persona_tone_fingerprint_rules():
    prompt = get_system_prompt()
    assert "1-3" in prompt or "1～3" in prompt
    assert "口癖" in prompt
    assert "0" in prompt and "2" in prompt  # 口癖 0-2


def test_persona_contradiction_traits():
    prompt = get_system_prompt()
    for trait in ("元气", "屑", "笨", "温柔", "倔"):
        assert trait in prompt


def test_persona_forbids_graduation_as_joke():
    prompt = get_system_prompt()
    assert "笑料" in prompt or "玩成笑料" in prompt
```

- [ ] **步骤 2：运行测试验证失败**

```bash
.venv/bin/python -m pytest tests/test_persona.py -v
```

预期：新断言 FAIL（当前 prompt 未锁 2020–2023 或未写口癖密度）。

- [ ] **步骤 3：按 `persona-spec.md` 重写 `HUA_YIN_SYSTEM_PROMPT`**

要求：

- 仍含：AI 复刻、2026-05-01、不制作周边、中文为主、禁止首先/其次/列表、禁止括号动作、毕业官方原因、私事回避  
- **新增：** 基准期 2020–2023；五特质+反例各一句；语气 6 条；口癖 0–2；场景压缩要点（不必每场景长文）  
- **删除/避免：** 大段时间线散文、官方 6 条全文堆进 prompt（留给 few-shot）  
- 长度目标：约 60–90 行中文，可读可维护  

`get_system_prompt()` 仍只返回该常量。

- [ ] **步骤 4：运行测试验证通过**

```bash
.venv/bin/python -m pytest tests/test_persona.py tests/test_persona_context.py -v
```

预期：全部 PASS（context 侧尚未改行为也应过）。

- [ ] **步骤 5：Commit**

```bash
git add dialogue/persona.py tests/test_persona.py
git commit -m "feat(persona): 按规格编译元气期人设卡与不变量测试"
```

---

### 任务 5：T1 测评（仅 prompt）

**文件：**
- 创建：`docs/persona/eval-results/T1-prompt.md`

- [ ] **步骤 1：同 T0 流程** 再跑 18 题（`asr_pool_enabled` 仍为 false）

- [ ] **步骤 2：记录金标+盲测**，与 T0 对比一段话（哪些题升/降）

- [ ] **步骤 3：Commit**

```bash
git add docs/persona/eval-results/T1-prompt.md
git commit -m "docs(persona): T1 测评（新 prompt，few-shot 未扩）"
```

---

### 任务 6：ASR → 审阅队列脚本

**文件：**
- 创建：`scripts/build_persona_review_queue.py`
- 创建：`tests/test_build_persona_review_queue.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_build_persona_review_queue.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

# 脚本在 scripts/ 下，测试里按项目惯例 import
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_persona_review_queue as bq  # noqa: E402


def test_normalize_text_collapses_space():
    assert bq.normalize_text("  阿里  嘎多  ") == "阿里嘎多"


def test_content_len_gate():
    assert bq.in_length_range("哈哈") is False  # 过短
    assert bq.in_length_range("这可是笨蛋尼们") is True


def test_guess_era_from_source_peak():
    assert bq.guess_era_tag("BV1xx_2022-05-01_杂谈.wav") == "peak"
    assert bq.guess_era_tag("【眞白花音】2026_04_15 白菜来啦.wav") == "late"


def test_build_queue_filters_and_caps(tmp_path: Path):
    asr = [
        {
            "path": str(tmp_path / "BV_2022_peak_(Vocals)__1.wav"),
            "lang": "zh",
            "text": "尼们现在是什么心情呀",
            "avg_logprob": -0.1,
            "language_probability": 0.95,
            "segment_count": 1,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.0,
        },
        {
            "path": str(tmp_path / "BV_2026_late_(Vocals)__2.wav"),
            "lang": "zh",
            "text": "谢谢大家今天也来看直播",
            "avg_logprob": -0.1,
            "language_probability": 0.95,
            "segment_count": 1,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.0,
        },
        {
            "path": str(tmp_path / "bad.wav"),
            "lang": "zh",
            "text": "啊",
            "avg_logprob": -0.1,
            "language_probability": 0.95,
            "segment_count": 1,
            "max_no_speech_probability": 0.1,
            "max_compression_ratio": 1.0,
        },
    ]
    # 写假 asr 文件
    asr_path = tmp_path / "asr.json"
    asr_path.write_text(json.dumps(asr, ensure_ascii=False), encoding="utf-8")
    seed_fewshot = {
        "asr_pool": [
            {
                "text": "摆在不能原谅披萨汉堡里面的菠萝",
                "lang": "zh",
                "avg_logprob": -0.15,
                "high_conf": True,
                "scenes": ["自嘲"],
                "source": "seed.wav",
            }
        ]
    }
    few_path = tmp_path / "few.json"
    few_path.write_text(json.dumps(seed_fewshot, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "queue.json"
    stats = bq.build_queue(
        asr_path=asr_path,
        fewshot_path=few_path,
        out_path=out,
        max_items=50,
        prefer_zh=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert stats["written"] >= 1
    texts = {item["text"] for item in data["items"]}
    assert "尼们现在是什么心情呀" in texts
    assert "啊" not in texts
    for item in data["items"]:
        assert item["status"] == "pending"
        assert item["era_tag"] in {"peak", "late", "unknown"}
```

- [ ] **步骤 2：运行测试验证失败**

```bash
.venv/bin/python -m pytest tests/test_build_persona_review_queue.py -v
```

预期：FAIL，无法 import 或缺函数。

- [ ] **步骤 3：实现 `scripts/build_persona_review_queue.py`**

最小实现要点：

```python
"""从 ASR 与现有 fewshot asr_pool 生成人设审阅队列（不注入运行时）。"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

# 允许 `python scripts/build_persona_review_queue.py` 直接跑
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_text_quality import content_chars, evaluate_text_quality  # noqa: E402

# high_conf 与 tests/test_persona_context.py 历史门槛对齐
def meets_asr_conf(record: dict) -> bool:
    return (
        record.get("segment_count") in (1, None)
        and (record.get("avg_logprob") or -99) >= -0.2
        and (record.get("language_probability") or 0) >= 0.9
        and (record.get("max_no_speech_probability") or 0) <= 0.3
        and (record.get("max_compression_ratio") or 99) <= 1.2
        and evaluate_text_quality(record.get("text") or "", record.get("lang")).keep
    )


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", "", t)


def in_length_range(text: str, lo: int = 4, hi: int = 40) -> bool:
    n = len(content_chars(text))
    return lo <= n <= hi


def guess_era_tag(source_name: str) -> str:
    """从文件名猜时代；无法判断则 unknown。

    注意：仓库内大量素材为 2025–2026，peak 可能偏少——队列仍应收 late，
    由人审决定是否 approved 及是否标 late。
    """
    years = [int(y) for y in re.findall(r"(20[0-2][0-9])", source_name or "")]
    # 过滤明显非活动年份噪声时可再收紧；优先看 2019-2026
    years = [y for y in years if 2019 <= y <= 2026]
    if not years:
        return "unknown"
    y = years[0]
    if 2020 <= y <= 2023:
        return "peak"
    if y >= 2024:
        return "late"
    return "unknown"


def build_queue(
    *,
    asr_path: Path,
    fewshot_path: Path,
    out_path: Path,
    max_items: int = 180,
    prefer_zh: bool = True,
) -> dict:
    asr = json.loads(asr_path.read_text(encoding="utf-8"))
    few = json.loads(fewshot_path.read_text(encoding="utf-8")) if fewshot_path.exists() else {}
    seen: set[str] = set()
    items: list[dict] = []

    def try_add(text: str, lang: str, source: str, avg_logprob, suggested_scenes=None, from_seed=False):
        nonlocal items
        if len(items) >= max_items:
            return
        text = (text or "").strip()
        if not text or not in_length_range(text):
            return
        key = normalize_text(text)
        if key in seen:
            return
        if not evaluate_text_quality(text, lang).keep:
            return
        if prefer_zh and lang == "ja" and not from_seed:
            # 日语降权：先装满中文后再考虑（实现时可二阶段）
            return
        seen.add(key)
        items.append(
            {
                "id": f"Q{len(items)+1:04d}",
                "text": text,
                "lang": lang or "zh",
                "source": source,
                "avg_logprob": avg_logprob,
                "suggested_scenes": suggested_scenes or [],
                "status": "pending",
                "final_text": "",
                "scenes": [],
                "era_tag": guess_era_tag(source),
                "reviewer_note": "",
            }
        )

    # 1) 现有 fewshot asr_pool 种子优先
    for row in few.get("asr_pool") or []:
        try_add(
            row.get("text", ""),
            row.get("lang", "zh"),
            row.get("source", "fewshot-seed"),
            row.get("avg_logprob"),
            row.get("scenes") or [],
            from_seed=True,
        )

    # 2) ASR high_conf：peak 优先排序
    candidates = []
    for rec in asr:
        if not meets_asr_conf(rec):
            continue
        source = Path(rec.get("path") or "").name
        candidates.append((guess_era_tag(source), rec, source))
    rank = {"peak": 0, "unknown": 1, "late": 2}
    candidates.sort(key=lambda x: (rank.get(x[0], 9), x[1].get("lang") != "zh"))

    for era, rec, source in candidates:
        if len(items) >= max_items:
            break
        try_add(rec.get("text", ""), rec.get("lang", "zh"), source, rec.get("avg_logprob"))

    # 3) 若中文不足 max 的 70%，二阶段允许 ja
    # （实现完整逻辑；测试用小数据不强制）

    out = {
        "schema_version": 1,
        "note": "待人审；不得被 build_persona_context 读取",
        "items": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"written": len(items)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asr", type=Path, default=Path("data/asr_results.json"))
    p.add_argument("--fewshot", type=Path, default=Path("docs/persona/fewshot-lines.json"))
    p.add_argument("--out", type=Path, default=Path("docs/persona/asr-review-queue.json"))
    p.add_argument("--max-items", type=int, default=180)
    args = p.parse_args(argv)
    stats = build_queue(
        asr_path=args.asr,
        fewshot_path=args.fewshot,
        out_path=args.out,
        max_items=args.max_items,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

实现时把注释中的「二阶段 ja」写成真实代码：中文写满后若 `len(items) < max_items`，再扫 ja 候选。

- [ ] **步骤 4：运行测试验证通过**

```bash
.venv/bin/python -m pytest tests/test_build_persona_review_queue.py -v
```

预期：PASS。

- [ ] **步骤 5：生成真实队列**

```bash
.venv/bin/python scripts/build_persona_review_queue.py --max-items 180
```

预期：生成 `docs/persona/asr-review-queue.json`，`items` 长度约 120–180（若 ASR 不足可更少，但应 > 40）。

```bash
python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('docs/persona/asr-review-queue.json').read_text())
print(len(d['items']))
assert len(d['items']) >= 40
print('ok')
PY
```

- [ ] **步骤 6：Commit**

```bash
git add scripts/build_persona_review_queue.py tests/test_build_persona_review_queue.py docs/persona/asr-review-queue.json
git commit -m "feat(persona): ASR 人设审阅队列生成脚本与首包候选"
```

若 `asr-review-queue.json` 过大或不想入库，可只 commit 脚本+测试，队列本地生成；**默认建议入库**便于续审。

---

### 任务 7：合并脚本 + 改写 asr 池质量测试

**文件：**
- 创建：`scripts/merge_persona_fewshot.py`
- 创建：`tests/test_merge_persona_fewshot.py`
- 修改：`tests/test_persona_context.py`（`test_asr_pool_quality_gate_reconnect`）
- 修改：`dialogue/persona_context.py`（late 过滤）

- [ ] **步骤 1：写 merge 失败测试**

`tests/test_merge_persona_fewshot.py`：

```python
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import merge_persona_fewshot as merge  # noqa: E402


def test_merge_only_approved_and_edited(tmp_path: Path):
    few = {
        "schema_version": 1,
        "note": "x",
        "injection_strategy": "y",
        "asr_pool_enabled": False,
        "official_comments": [{"text": "阿里嘎多！红豆泥阿里嘎多", "lang": "zh", "scenes": ["感谢"]}],
        "fan_pool": [],
        "asr_pool": [{"text": "旧的未审", "lang": "zh", "scenes": [], "source": "old.wav"}],
    }
    queue = {
        "schema_version": 1,
        "items": [
            {
                "id": "Q1",
                "text": "坏的",
                "final_text": "尼们好呀",
                "lang": "zh",
                "source": "a.wav",
                "status": "edited",
                "scenes": ["问候"],
                "era_tag": "peak",
            },
            {
                "id": "Q2",
                "text": "找到蟑螂了",
                "final_text": "",
                "lang": "zh",
                "source": "b.wav",
                "status": "approved",
                "scenes": ["屑"],
                "era_tag": "peak",
            },
            {
                "id": "Q3",
                "text": "不要",
                "final_text": "",
                "lang": "zh",
                "source": "c.wav",
                "status": "rejected",
                "scenes": ["问候"],
                "era_tag": "peak",
            },
            {
                "id": "Q4",
                "text": "毕业了哈哈",
                "final_text": "",
                "lang": "zh",
                "source": "d.wav",
                "status": "approved",
                "scenes": ["告别"],
                "era_tag": "late",
            },
        ],
    }
    few_path = tmp_path / "few.json"
    q_path = tmp_path / "q.json"
    few_path.write_text(json.dumps(few, ensure_ascii=False), encoding="utf-8")
    q_path.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out.json"
    stats = merge.merge_fewshot(few_path, q_path, out, enable_if_ready=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    texts = [x["text"] for x in data["asr_pool"]]
    assert "尼们好呀" in texts
    assert "找到蟑螂了" in texts
    assert "坏的" not in texts
    assert "不要" not in texts
    assert "旧的未审" not in texts
    assert data["asr_pool_enabled"] is False
    late = [x for x in data["asr_pool"] if x.get("era_tag") == "late"]
    assert len(late) == 1
    assert stats["approved_count"] == 3
```

- [ ] **步骤 2：运行确认 FAIL**

```bash
.venv/bin/python -m pytest tests/test_merge_persona_fewshot.py -v
```

- [ ] **步骤 3：实现 `scripts/merge_persona_fewshot.py`**

要点：

- 只合并 `status in {approved, edited}`  
- `edited` 用 `final_text`（非空），否则用 `text`  
- 必须有非空 `scenes`，否则跳过并计入 `skipped_no_scene`  
- 输出 asr_pool 条目字段：`text, lang, scenes, source, era_tag, high_conf: true`  
- `enable_if_ready=True` 时：若 approved 总数 ≥40 且 问候/夸赞/自嘲/食物/游戏 各 ≥3，则设 `asr_pool_enabled=true`  
- 保留 official_comments / fan_pool / note / injection_strategy  

- [ ] **步骤 4：改写 `test_asr_pool_quality_gate_reconnect`**

**旧逻辑错误地要求「所有 high_conf 都在 pool」**（全量导出假设）。新人设流程 pool 是**人审子集**。改为：

```python
def test_asr_pool_entries_pass_quality_when_enabled_sources_exist():
    """asr_pool 为人审子集：每条须有 scenes；若能回连 ASR 则过质量门。"""
    from scripts.dataset_text_quality import evaluate_text_quality

    data = json.loads(
        (ROOT / "docs/persona/fewshot-lines.json").read_text(encoding="utf-8")
    )
    pool = data.get("asr_pool") or []
    if not pool:
        pytest.skip("asr_pool 为空（尚未人审合并）")
    asr_path = ROOT / "data/asr_results.json"
    by_name = {}
    if asr_path.exists():
        asr = json.loads(asr_path.read_text(encoding="utf-8"))
        by_name = {Path(r["path"]).name: r for r in asr}

    for item in pool:
        assert (item.get("text") or "").strip()
        assert item.get("scenes"), item
        assert evaluate_text_quality(item["text"], item.get("lang")).keep
        src = item.get("source") or ""
        name = Path(src).name
        if name in by_name:
            rec = by_name[name]
            # 人审改写文本时可能与 ASR 原文不同，只校验源文件曾存在
            assert rec is not None
```

删除 `leaked == 0` 全量闭合断言。

- [ ] **步骤 5：`persona_context` late 过滤**

在 `_pick_fewshot` 中，对 asr_pool 条目：

- 若 `era_tag == "late"` 且当前匹配场景**不在** `{"告别"}`，则跳过该条  
- 无 `era_tag` 字段视为可注入（兼容旧数据）  

追加测试到 `tests/test_persona_context.py`：

```python
def test_fewshot_skips_late_unless_farewell_scene():
    data = _synthetic_lines()
    data["asr_pool_enabled"] = True
    data["official_comments"] = []
    data["fan_pool"] = []
    data["asr_pool"] = [
        {"text": "晚后期温柔台词", "lang": "zh", "scenes": ["问候"], "era_tag": "late"},
        {"text": "元气问候呀", "lang": "zh", "scenes": ["问候"], "era_tag": "peak"},
    ]
    picked = _pick_fewshot("你好呀", data, random.Random(1))
    assert "元气问候呀" in picked
    assert "晚后期温柔台词" not in picked

    data["asr_pool"] = [
        {"text": "好好休息哦", "lang": "zh", "scenes": ["告别"], "era_tag": "late"},
    ]
    picked2 = _pick_fewshot("毕业了好想她", data, random.Random(1))
    assert "好好休息哦" in picked2
```

- [ ] **步骤 6：跑测**

```bash
.venv/bin/python -m pytest tests/test_merge_persona_fewshot.py tests/test_persona_context.py -v
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add scripts/merge_persona_fewshot.py tests/test_merge_persona_fewshot.py \
  tests/test_persona_context.py dialogue/persona_context.py
git commit -m "feat(persona): few-shot 合并脚本、late 过滤与 asr 子集质量测试"
```

---

### 任务 8：人审 50–80 条并启用 asr 池

**文件：**
- 修改：`docs/persona/asr-review-queue.json`（status/scenes/final_text）
- 修改：`docs/persona/fewshot-lines.json`

- [ ] **步骤 1：人审**

打开队列，按设计规格采纳标准处理，目标：

- `approved` + `edited` ≥ **50**（冲刺 50–80）  
- 核心五场景各 ≥3：问候、夸赞、自嘲、食物、游戏  
- 优先 peak；late 仅在台词确实只适合告别时保留  

可分多 session；每批 20 条也可。

- [ ] **步骤 2：合并（先不自动 enable）**

```bash
.venv/bin/python scripts/merge_persona_fewshot.py \
  --fewshot docs/persona/fewshot-lines.json \
  --queue docs/persona/asr-review-queue.json \
  --out docs/persona/fewshot-lines.json
```

- [ ] **步骤 3：检查门槛后 enable**

```bash
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
d=json.loads(Path('docs/persona/fewshot-lines.json').read_text())
pool=d.get('asr_pool') or []
print('count', len(pool))
c=Counter()
for i in pool:
    for s in i.get('scenes') or []:
        c[s]+=1
print(dict(c))
core=['问候','夸赞','自嘲','食物','游戏']
ok=len(pool)>=40 and all(c[s]>=3 for s in core)
print('ready', ok)
PY
```

若 `ready True`，将 `asr_pool_enabled` 设为 `true`（merge 脚本 `--enable-if-ready` 或手改 JSON）。

- [ ] **步骤 4：回归测试**

```bash
.venv/bin/python -m pytest tests/test_persona.py tests/test_persona_context.py -v
```

- [ ] **步骤 5：Commit**

```bash
git add docs/persona/asr-review-queue.json docs/persona/fewshot-lines.json
git commit -m "feat(persona): 人审 few-shot 入库并启用 asr 池"
```

若未达门槛：commit 已审部分，**保持** `asr_pool_enabled: false`，继续人审；不得强行开启。

---

### 任务 9：事实库小补（可选，0–8 条）

**文件：**
- 修改：`docs/persona/facts-kb.json`（仅当 T1/T2 暴露事实空洞）

- [ ] **步骤 1：对照 E12/E13 与高频失败题**，仅当 RAG 未命中且影响语气时新增

示例 id 从 `F031` 起；`cat` 用现有枚举；敏感内容用 `sensitive`。

- [ ] **步骤 2：**

```bash
.venv/bin/python -m pytest tests/test_persona_context.py -v
```

- [ ] **步骤 3：Commit**（若有改动）

```bash
git add docs/persona/facts-kb.json
git commit -m "docs(persona): 事实库小补以支撑语气相关问答"
```

无改动则跳过本任务。

---

### 任务 10：T2 终测与收尾

**文件：**
- 创建：`docs/persona/eval-results/T2-final.md`
- 可选：`scripts/run_persona_eval.py`（仅 must_not 扫描 + 若配置了 LLM 则批量拉回复）

- [ ] **步骤 1：跑满 18 题**（asr 已启用）

- [ ] **步骤 2：金标 + 盲测**，对照完成线：

| 指标 | 需要 |
|------|------|
| 金标通过率 | ≥ 80% |
| E14–E16 | 100% |
| E17–E18 | 100% |
| 盲测均分 | ≥ 2.4 |
| 盲测 1 分题 | 0 |

- [ ] **步骤 3：未达标时的处理顺序（写在 T2 文档）**

1. 换/增 few-shot（同场景）  
2. 收紧 `persona.py` 指纹一句  
3. 仍失败再关 `asr_pool_enabled` 对比是否 asr 噪声  

不得用「降低完成线」交差。

- [ ] **步骤 4：可选 eval 脚本**

若实现 `scripts/run_persona_eval.py`：读取 eval-set、对每条回复扫 `must_not`、打印未过 id；**不**自动判语义 checklist。

- [ ] **步骤 5：全量单测**

```bash
.venv/bin/python -m pytest tests/test_persona.py tests/test_persona_context.py \
  tests/test_build_persona_review_queue.py tests/test_merge_persona_fewshot.py -v
```

- [ ] **步骤 6：Commit**

```bash
git add docs/persona/eval-results/T2-final.md
git commit -m "docs(persona): T2 终测记录（金标+盲测双过）"
```

仅当双过完成线后，在 T2 文档末行写：`STATUS: PASSED`。

---

## 规格覆盖自检

| 规格章节 | 任务 |
|----------|------|
| §2 资产分层 | 任务 1–2、6–8 |
| §3 人设规格与编译 | 任务 1、4 |
| §4 语料流水线 | 任务 6–8 |
| §5 验收 T0/T1/T2 | 任务 3、5、10 |
| §6 降级 / late | 任务 7 |
| §7 实现顺序 | 任务 1→10 |
| §8 不做清单 | 全任务未引入向量库/微调/全量注入 |

## 占位符扫描

计划内步骤均含具体路径、命令或代码；人审步骤为人工操作但有数量门槛与合并命令，无「TODO/待定」实现空洞。

---

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-08-10-baicai-persona-restoration.md`。

**两种执行方式：**

1. **子代理驱动（推荐）** — 每个任务调度一个新子代理，任务间审查，快速迭代  
2. **内联执行** — 在当前会话用 executing-plans 批量执行并设检查点  

**选哪种方式？**
