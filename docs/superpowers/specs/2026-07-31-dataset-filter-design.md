# 训练数据过滤管线设计

> **日期：** 2026-07-31
> **状态：** 已实现并验收（含声纹 accept 阶段）
> **关联：** `docs/superpowers/specs/2026-07-30-mashiro-huayin-tts-design.md` §3.2

## 背景

现有数据管线 `scripts/build_dataset.py` 已产出 195 条训练数据，但存在严重质量问题：

1. **素材来源偏差**：已处理的 4 个 WAV 全部是"白菜看…"反应切片，混入了大量被看视频的原声（UVR5 BS-Roformer 只分离 BGM，分不开双人声）。
2. **歌声混入**：花音在直播中经常唱歌，歌声的发声方式与说话不同，混入训练集会损害 TTS 模型的说话质量。
3. **Whisper 幻觉**：在歌声/噪音频段，faster-whisper 产生语种误判（出现 vi/es/ko/nn 等异常）和乱码文本。

当前 195 条数据集不可用，需要重建。

## 设计目标

- **素材层面**：只用 17 个直播杂谈类文件（花音说话为主），排除 3 个纯歌唱类和 8 个反应切片类。
- **片段层面**：用音频特征自动检测并剔除歌声段、静音/噪音段。
- **文本层面**：用 ASR 质量信号（语种、文本质量）剔除幻觉转写。
- **渐进式**：先做启发式过滤（不安装新依赖），检查效果后再决定是否加声纹验证层。

## 素材筛选

### 三类素材

| 类型 | 数量 | 特点 | 处理 |
|------|------|------|------|
| 纯歌唱（演唱会/歌回） | 3 | 几乎全是歌声 | 排除 |
| 反应切片（"白菜看…"） | 8 | 混入被看视频的原声 | 排除 |
| 直播杂谈（"白菜来啦"/录播） | 17 | 花音说话为主，偶有游戏声/短歌声 | 纳入 |

### 白名单机制

新增 `data/training_assets.txt`，逐行列出纳入训练的 WAV 文件名（不含路径）。separate_vocals() 读取白名单，仅处理其中的文件。不在白名单中的文件完全跳过。

判定某文件属于哪一类，按标题关键词分类（在实现时生成初始白名单并人工校验）：
- **排除**：标题含"歌""演唱会""歌回"（歌唱类），或"看"且非"录播"上下文（反应切片类）
- **纳入**：标题含"白菜来啦""直播""录播""菜鸟""小游戏""逃离鸭科夫"等杂谈/游戏关键词

## 管线架构

```
现有:  separate → slice → asr → dataset
改为:  separate → slice → filter → asr → dataset
                         ^^^^^^
                         新增阶段
```

### 各阶段职责

| 阶段 | 输入 | 输出 | 职责 |
|------|------|------|------|
| separate | data/wav/ (白名单) | data/vocals/ | UVR5 人声分离（去 BGM），仅处理白名单文件 |
| slice | data/vocals/ | data/slices/ | RMS 静音切分为 5-15s 短句 |
| **filter** | data/slices/ | data/slices/ (标注) | 歌声检测 + 能量/时长过滤，生成过滤标注 |
| asr | data/slices/ (过滤后) | data/asr_results.json | faster-whisper 转写（增量） |
| dataset | asr_results.json | data/dataset/ | 语种/文本质量过滤 + 汇总训练集 |

filter 阶段放在 ASR 之前：先用音频特征剔除明显的不合格片段，减少 ASR 处理量（ASR 是最慢的阶段）。文本质量过滤放在 dataset 阶段，因为需要 ASR 结果作为信号。

## 歌声检测算法（filter 阶段）

### 原理

歌声与说话在基频行为上有显著差异：
- **歌声**：持续发声（voiced_ratio 高），音高稳定（F0 变异系数低），有长持续音
- **说话**：有停顿（voiced_ratio 中等），音高变化大（F0 变异系数高）， segment 短

### 实现

对每个切片用 `librosa.pyin` 提取 F0 轨迹（fmin=80, fmax=400，覆盖人声范围），计算：

1. **voiced_ratio** = voiced 帧数 / 总帧数
2. **f0_cv**（变异系数）= 最长连续 voiced 段内 F0 的 std / |mean|
3. **longest_voiced** = 最长连续 voiced 段时长（秒）

### 判定规则

标记为歌声并剔除，满足以下任一：
- voiced_ratio > 0.75 且 f0_cv < 0.15（持续稳定发声 = 唱歌）
- longest_voiced > 3.0s（单次连续发声超 3 秒，说话极少出现）

同时剔除：
- **能量过低**：RMS 均值 < -40dB（静音/极低音量 = 噪音）
- **时长异常**：< 1.5s 或 > 30s（太短无意义/太长可能含多句）

### 输出

filter 阶段不删除文件，而是生成标注文件 `data/filter_results.json`：
```json
[
  {"path": "data/slices/xxx.wav", "keep": false, "reason": "singing", "voiced_ratio": 0.91, "f0_cv": 0.08},
  {"path": "data/slices/yyy.wav", "keep": true, "reason": null, "voiced_ratio": 0.52, "f0_cv": 0.31}
]
```

ASR 阶段读取此文件，只转写 `keep: true` 的切片。dataset 阶段也据此过滤。

### 阈值调试

初始阈值基于上述经验值。实现时先对已有切片跑一遍 filter，抽样检查被判为歌声和保留的片段，确认判定合理后再全量运行。阈值全部参数化（命令行 `--voiced-ratio-threshold` 等），便于调整。

## ASR 质量过滤（dataset 阶段）

在 build_dataset() 汇总训练集时，除现有的文本长度/文件大小过滤外，增加：

1. **语种过滤**：只保留 ja/zh/en（花音实际使用的语言），映射到 GPT-SoVITS 标签（JP/ZH/EN）。剔除 nn（无语言检测）、vi/es/ko/"yue" 等异常语种——这些是 Whisper 幻觉的强信号。
2. **重复字符过滤**：文本中同一字符重复占比 > 50% 时剔除（如"啊啊啊啊啊"型幻觉）。

## 增量处理修复

### slice 阶段（当前有 bug）

**现状**：`if existing: return` — 只要 slices 目录有文件就全部跳过，新 vocal 不会被切分。

**修复**：改为按 vocal 文件逐个增量检查。对每个 vocal，检查是否已有 `"{stem}__*.wav"` 切片；有则跳过该 vocal，无则切分。

### ASR 阶段

**现状**：每次全量转写所有切片，覆盖 asr_results.json。

**修复**：加载已有 asr_results.json，跳过已转写的切片路径，只转写新的，合并写回。

### filter 阶段

天然增量：对已有 filter_results.json 中的切片跳过，只分析新切片。

## 数据重建

清除旧的反应切片产物，从 17 个杂谈类重新走管线：

1. 清空 `data/vocals/`（4 个反应切片的人声分离结果）
2. 清空 `data/slices/`（218 个切片）
3. 删除 `data/asr_results.json`
4. 清空 `data/dataset/`（195 条训练数据）
5. 生成 `data/training_assets.txt`（17 个杂谈类白名单）
6. 运行 `python scripts/build_dataset.py --stage all`

## 外部人声的后续方案（Phase 2，视效果决定）

启发式过滤对歌声和噪音有效，但对**同为人声的外部声音**（游戏 NPC 语音、连麦联动等）识别力有限。ASR 语种/质量过滤能间接剔除一部分（非花音声音的转写质量通常更差）。

如果 Phase 1 完成后数据集仍有明显的外部人声残留，启动 Phase 2：
- 安装 speechbrain（ECAPA-TDNN 声纹模型）
- 从杂谈素材中截取花音纯净说话片段作为参考音频
- 对每个切片做声纹比对，剔除与花音声纹不匹配的片段

## 测试策略

- **歌声检测单元测试**：构造已知特征的合成信号（稳定正弦波 = 歌声，随机调制 = 说话），验证判定逻辑
- **ASR 过滤单元测试**：给定模拟的 asr_results（含各种语种/文本质量），验证过滤结果
- **增量逻辑测试**：验证 slice/asr 在部分完成时能正确续跑
- **集成验证**：对一小批杂谈素材跑完整管线，人工抽检过滤效果

## 文件变更

| 文件 | 变更 |
|------|------|
| `scripts/build_dataset.py` | 新增 filter 阶段、白名单读取、增量修复、ASR 质量过滤增强 |
| `data/training_assets.txt` | 新增：17 个杂谈类白名单 |
| `tests/test_build_dataset.py` | 新增：filter/增量/质量过滤的单元测试 |

## 验收记录（2026-08-06）

- training_assets.txt 当前 22 个白名单素材（在初版 17 个杂谈素材基础上纳入了已审计的完整候选）；分离、切片、特征过滤、声纹和 ASR 均支持增量缓存。
- filter_results.json：31,463 条切片，保留 18,075，歌声 10,749，时长异常 2,285，低能量 354。
- speaker_results.json：18,075 条，accept 16,028，uncertain 2,047；最终数据集只消费 accept。
- asr_results.json：16,399 条，结果由 build_dataset 按语种、置信度、文本质量和文件存在性汇总。
