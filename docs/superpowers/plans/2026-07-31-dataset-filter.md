# 训练数据过滤管线实现计划

> **规格：** `docs/superpowers/specs/2026-07-31-dataset-filter-design.md`
>
> **执行方式：** 在当前会话中按 TDD 小步实现；每个任务先写失败测试，再写最小实现，最后运行相关测试。

**目标：** 将现有 `separate → slice → asr → dataset` 管线扩展为带素材白名单、歌声/低能量过滤、ASR 质量过滤和断点续跑能力的可靠训练数据管线。

**约束：** 保留用户已有的 `librosa` 依赖改动；`librosa` 必须懒加载，避免不运行 filter 阶段时阻断其他命令；不引入 Phase 2 的声纹依赖。

---

## 任务 1：建立测试基线与素材白名单

**文件：**

- 修改：`.gitignore`
- 新增：`data/training_assets.txt`
- 新增：`tests/test_build_dataset.py`

1. 让 `data/training_assets.txt` 成为 `data/` 下唯一可跟踪的配置文件。
2. 将 28 个 WAV 按规格分为 17 个纳入、8 个反应素材排除、3 个歌唱素材排除。
3. 先写白名单读取测试：只返回白名单中的 WAV、空行可忽略、缺失文件给出明确错误。
4. 运行该测试并确认因功能缺失而失败。

## 任务 2：实现白名单分离与逐文件增量切片

**文件：**

- 修改：`scripts/build_dataset.py`
- 修改：`tests/test_build_dataset.py`

1. 恢复当前未完成改动误删的标准库导入，并将第三方音频库保持为函数内懒加载。
2. 实现 `load_training_assets()`，`separate_vocals()` 仅处理白名单文件。
3. 用文件名 `startswith` 判断已有产物，避免标题中的 `[]` 被 `glob` 当作模式字符。
4. 为 `slice_audio()` 写部分完成测试：已有 A 的切片时跳过 A，仍切分 B。
5. 实现逐 vocal 增量逻辑，并运行相关测试。

## 任务 3：实现音频特征过滤阶段

**文件：**

- 修改：`scripts/build_dataset.py`
- 修改：`tests/test_build_dataset.py`

1. 写纯判定逻辑测试，覆盖稳定高占比 F0、超长连续发声、低能量、过短/过长和正常说话。
2. 写合成信号测试：稳定正弦波应呈现歌声特征；带停顿的调频信号应呈现更短连续发声和更高 F0 变化。
3. 实现 `librosa.pyin` 特征提取：`voiced_ratio`、最长 voiced 段的 `f0_cv`、`longest_voiced`、RMS dB 和时长。
4. 实现增量 `filter_slices()`，输出 `data/filter_results.json`，保留文件而只写标注。
5. 添加 `filter` CLI stage、全部阈值参数和 `--force-refilter` 调试开关。

## 任务 4：实现过滤感知的增量 ASR

**文件：**

- 修改：`scripts/build_dataset.py`
- 修改：`tests/test_build_dataset.py`

1. 写测试证明 ASR 只处理 `keep: true` 且尚未缓存的切片。
2. 写测试证明没有新切片时不会加载 Whisper 模型。
3. 加载已有 `asr_results.json`，兼容相对/绝对路径，合并新结果后原子写回。
4. 在 `all` 流程中确保顺序为 `separate → slice → filter → asr → dataset`。

## 任务 5：实现 dataset 质量过滤

**文件：**

- 修改：`scripts/build_dataset.py`
- 修改：`tests/test_build_dataset.py`

1. 写测试覆盖 `ja/zh/en` 映射及 `nn/vi/es/ko/yue` 排除。
2. 写测试覆盖同一非空白字符占比大于 50% 的幻觉文本排除。
3. 写测试覆盖 filter 标记为 false、缺失文件、短文本和极短音频排除。
4. 实现汇总逻辑并输出按原因统计的过滤结果。

## 任务 6：验证、阈值抽检与交付

1. 运行 `tests/test_build_dataset.py`，再运行完整测试集。
2. 运行 `python scripts/build_dataset.py --help` 和编译检查，确认无顶层可选依赖阻断。
3. 对现有切片运行 filter，检查保留/剔除计数及两类样本特征；若证据表明阈值明显偏离，再调整并重跑。
4. 在确认实现和阈值后，按规格处理旧生成物；若全量 UVR5/ASR 运行成本超出当前验证窗口，保留可恢复的数据并给出精确续跑命令和状态。
5. 做一次独立代码复核，检查规格覆盖、路径兼容、增量幂等和用户已有改动是否完整保留。

## 执行记录（2026-08-03）

- 完整性审计后，初始训练白名单收敛为 8 场完整、去重的纯杂谈倾向素材，共 22.29 小时；原 17 项中 7 项严重截短，重复/小游戏素材也已移除。
- 额外找到 14 个尚未下载的纯杂谈候选。当前新增 `BV1WQ9YBNEAp` 已完整下载并通过整条解码检查，待首批数据集生成后增量纳入。
- 运行机有两张 RTX 3090，但主存仅 31.2 GiB。两个数小时 WAV 直接并行分离会把 worker 以 `-9` 杀死；`audio_separator.Separator` 必须设置 `chunk_duration=1800` 和 `output_single_stem="Vocals"`，再按剩余音频总时长均衡到两张 GPU。
- 后续阶段使用 4 个 CPU 进程切片、12 个 CPU 进程提取音频特征、两张 GPU 并行 ASR。只保留少量关键测试，主要验证实际素材产出。
