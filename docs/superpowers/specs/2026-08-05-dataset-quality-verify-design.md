# 数据集音文一致性与文本质量校验设计

> **日期：** 2026-08-05  
> **状态：** 已实现并验收  
> **关联：** `docs/superpowers/specs/2026-07-31-dataset-filter-design.md`

## 背景

现有管线已产出约 1.2 万段训练数据（声纹 accept + ASR 置信度过滤），但仍存在：

1. **文本异常**：短语循环（「やばいやばい…」）、字符狂奔、纯填充音
2. **语种标签与文字脚本不一致**：标成 ja/zh 但正文几乎全是拉丁字母或其它脚本
3. **音文不一致**：Whisper 幻觉——音频是花音说话，转写却是乱码/错语种/重复噪声

仅靠 `language_probability` / `avg_logprob` 无法覆盖上述问题。

## 目标

- 对中日语料做**可解释、可复现**的二次清洗
- 校验「文本是否正常」与「语音是否像对应文本」
- 输出可审计的中间结果，并重建 `data/dataset/`

## 方案（两层）

### 层 1：确定性文本质量门（CPU，秒级）

模块：`scripts/dataset_text_quality.py`

| 检查 | 说明 |
|------|------|
| script_ratio | 正文中与声称语种匹配的字符占比（ja: 假名+汉字，zh: 汉字） |
| phrase_loop | 连续短语循环（2–8 字单元重复 ≥3 次） |
| repetitive_char / char_run | 单字占比过高或 5+ 连续相同字符 |
| hangul | 韩文污染 |
| latin_heavy | 中日条目拉丁字母占比过高 |
| nonlexical | 去掉标点/拖长音后几乎无内容 |
| en_disabled | 默认训练集不收 EN（花音以中日为主） |

接入：`build_dataset.build_dataset()` 在 ASR 置信度过滤之外额外执行。

### 层 2：强制语种二次 ASR 对齐（GPU，小时级）

脚本：`scripts/verify_asr_alignment.py`

1. 候选 = filter keep ∧ speaker accept ∧ 文本门通过 ∧ ja/zh  
2. 对每条用 faster-whisper **强制** `language=ja|zh` 再转写  
3. 计算 `similarity_ratio(original, forced)`（SequenceMatcher）  
4. 判定：
   - forced 空 / 文本门失败 / logprob 差 / no_speech 高 → drop  
   - similarity < 0.45 → drop（音文严重不一致）  
   - 否则 keep；中等相似度且 forced 脚本更贴语种时，可用 forced 文本替换  

输出：`data/alignment_results.json`（增量）  
消费：`build_dataset --require-alignment`

## 重建命令

```bash
# 1) 仅文本门快速清洗（默认去 EN）
python scripts/build_dataset.py --stage dataset

# 2) 双卡音文对齐（可断点续跑）
python scripts/verify_asr_alignment.py --gpus 0 1 --min-similarity 0.45

# 3) 对齐后严格重建
python scripts/build_dataset.py --stage dataset --require-alignment
```

## 成功标准

- 文本门单测覆盖主要幻觉模式  
- 清洗后 annotation 无 hangul / 明显 phrase_loop / 脚本错配样本  
- 对齐完成后 `require_alignment` 训练集规模与 drop 原因可统计  
- 不破坏 filter / speaker / asr 增量缓存

## 验收记录（2026-08-06）

- 文本门测试和实际 annotation 复核均通过；data/dataset/annotation.list 共 9,876 条，逐条复核违规为 0。
- alignment_results.json 共 11,229 条：keep 9,876，drop 1,353；drop 原因可按 low_similarity、forced_logprob、文本门原因等统计。
- build_dataset --require-alignment 已消费 alignment 缓存并产出当前训练集，未重置 filter、speaker 或 ASR 缓存。
