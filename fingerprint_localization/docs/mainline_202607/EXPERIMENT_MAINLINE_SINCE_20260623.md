# 2026-06-23 以来实验主线与交接说明

更新日期：2026-07-12

本文是搭档接续对比实验的首要入口。论文结果以
`fingerprint_localization/experiments/aco_source_safe_1to10/MAINLINE_20260710_LOCK.md` 为唯一主线锁定依据；
`fingerprint_localization/experiments/aco_source_safe_1to10/data` 和 `results` 下的旧 6/2/2 结果仅作历史记录。

## 1. 一句话主线

离线用宽带 chirp 构建位置级多径先验和模板可靠性，在线只使用
LoRa packet 的 RSSI、raw-bin 和分段频谱观测；RSSI 先产生 Top-5 候选，
ACO v4 再进行分段路径搜索和 Score4 融合。最终采用 source-packet-safe
协议，独立 validation 选出 `top_k=5, rssi_class_k=3`，随后 train+val
refit，在 74 个未增强且与训练源零重叠的 test packet 上得到
`67/74 = 90.54%`。

## 2. 时间线

| 日期 | 实验推进 | 结论与主线地位 |
| --- | --- | --- |
| 06-23 | 从原始采集重建 54 点 RSSI+/LoRa 特征；对 28 个严格配对点完成 RSSI、S17、chirp step1-6 分析 | RSSI+ 约 73.93%；加入 `log(raw center bin)` 达 82.81%。S17 归一化形状直接融合无效。chirp 24 个文件中 8 个 fail，16 个有效实测点；最终保留 `step6c ... original_minus25` 结构。点 14 修正为 OLOS。 |
| 06-24 | q=1/q=4 zero-padding FFT、稳定性、t-SNE、Top-k 与混淆分析 | 368 包/30 点上 RSSI+ 72.55%，RSSI+ raw bins[-2,+2] 80.43%；q4 单独约 9.51%，直接拼接仅 42.39%，不应作为强分类特征。 |
| 06-25 | PGAR、初版 packet-internal ACO、ACO v2、MFR-ACO、Bayesian inversion | PGAR 82.34%，初版 ACO 79.08%，ACO v2 80.43%；MFR/Bayesian 未胜出。这一批是方法筛选，不是最终结果。 |
| 06-26 | 1:10 高斯噪声增强；chirp 到 LoRa bin 投影 | 旧协议先增强再切分，test 上 KNN 77.43%、PGAR 85.68%、ACO 87.03%，后确认存在同源泄漏，仅作 historical。chirp/LoRa 高形状相关后来被负对照证实主要来自普适主瓣形状。 |
| 06-27 | ACO v2.0 与 v4.0 主干形成 | v2 加入 RSSI Top-k、4 segment、chirp shrinkage、garbage state、动态切换惩罚；v4 加入弱先验、稳定性门控、cost veto、可靠性信息素和 Score4。 |
| 06-28 | v4.2-v4.8 候选纠错分支和稳定性实验 | v4.7/v4.8 在旧协议上较高，但属于比较/探索分支，不能替代最终 refit 主线。 |
| 06-30 | 时延评测 | 冻结后处理约 1.17 us/sample，缓存专家流水线约 295 us/sample；口径不同，且都不是最终 74 包 refit 的完整端到端时延。 |
| 07-10 | 协议纠偏、validation 选参、主线锁定 | 改为 source-first：train/val/test 源包 223/73/74，源交集为 0。val 选出 v4 Top5-k3；train+val 共 296 源包 refit 后 test 67/74。 |
| 07-10 至 07-11 | chirp-LoRa matched/mismatched、距离矩阵、bin-window、q4 complex 负对照 | 15 点上无显著位置特异直接匹配；matched top1 多为 1/15，median rank 约 8-10。不能声称 chirp 直接预测位置 LoRa 谱。 |
| 07-11 | 低阶物理结构解释 | chirp 稳定路径映射位移都落在 0.25 LoRa bin 内，LoRa 无法分辨独立路径。secondary shoulder 与多径强度只有探索性相关，多重校正不显著。 |
| 07-12 | 算法文档与 no-chirp 消融 | LoRa-only test 为 66/74，完整主线为 67/74；chirp 是离线弱先验，贡献 +1 包，不是在线输入。 |

## 3. 数据协议和口径

论文主线只能使用下面的 source-safe 流程：

1. 以 `(source_file_stem, source_packet_index)` 为 source ID。
2. seed 固定为 `20260626`，先划分 223/73/74 个 train/val/test 原始源包。
3. 只有 train source 生成 1:10 高斯增强；独立 val/test 保持原包。
4. 只在 73 个独立 val 包上选择配置，不按 test 结果选参。
5. 配置锁定后合并 train+val 的 296 个源包，得到 2960 个增强训练样本。
6. 最终 test 是 74 个原始包，不增强，与 refit train 的 source overlap 为 0。

注意：refit 数据中的 73 条 `val` 只为诊断保留，其 source 已进入 refit train，
因此不再是独立 validation，也不能把其 `70/73` 当作泛化成绩。

文件名中的 `54points` 表示原始候选地图口径，并不代表每一步都有 54 个有效位置：
LoRa 主实验对齐后为 32 个标签；中间物理分析常为 28/30 点；最终 74 个 test
packet 覆盖 30 个位置。chirp 只有 16 个有效实测位置，其余使用 nearest、
interpolation 或 fallback，不能把这些位置表述为实测 chirp。

## 4. 冻结结果

| 方法/协议 | test correct | test accuracy | 用途 |
| --- | ---: | ---: | --- |
| RSSI+ KNN，source-safe | 50/74 | 67.57% | 传统基线 |
| ACO v2.0，source-safe old split | 63/74 | 85.14% | ACO 基线 |
| ACO v4 top3-k3，source-safe old split | 63/74 | 85.14% | refit 前参考 |
| ACO v4.7 two-stage，source-safe old split | 62/74 | 83.78% | 比较分支，未 refit |
| ACO v4 top5-k3 + trainval refit | **67/74** | **90.54%** | 最终主结果 |
| ACO v4 top5-k3 LoRa-only + refit | 66/74 | 89.19% | chirp 消融 |

最终结果证据：

- `fingerprint_localization/experiments/aco_source_safe_1to10/group_safe_trainval_refit/results/refit_comparison_summary.json`
- `fingerprint_localization/experiments/aco_source_safe_1to10/group_safe_trainval_refit/results/aco_v4_top5_k3_refit/aco_v4_summary.csv`
- `fingerprint_localization/experiments/aco_source_safe_1to10/group_safe_trainval_refit/results/aco_v4_top5_k3_refit/test_predictions.csv`

## 5. 从仓库根目录复现

主线 ACO 代码仅依赖 Python 标准库。先准备
`docs/DATA_MANIFEST.csv` 中标为 `git` 或 `release` 的输入，然后执行：

```bash
python3 -B scripts/verify_handoff.py --hash
```

该命令检查清单 SHA-256、source overlap、样本数和冻结的 `67/74` 结果。
数据无误后再执行：

```bash
BASE=fingerprint_localization/experiments/aco_source_safe_1to10

# 生成 1:10 noisy pool。该脚本还会写出旧泄漏协议结果，不能用于论文主结果。
python3 -B "$BASE/run_experiment.py"

# source-first 独立 train/val/test
python3 -B "$BASE/build_group_safe_1to10_data.py"

# 从相同 source split 精确重建最终 train+val refit 数据
python3 -B "$BASE/build_trainval_refit_data.py"
```

ACO v2 source-safe 基线：

```bash
python3 -B "$BASE/run_aco_v2_on_split.py" \
  --result-dir "$BASE/group_safe_1to10/results" \
  --output-dir "$BASE/group_safe_1to10/results/aco_v2" \
  --method-summary "$BASE/group_safe_1to10/results/method_summary_with_aco_v2.csv" \
  --rssi-csv "$BASE/group_safe_1to10/data/noisy_rssi_plus_packet_level_54points.csv" \
  --spectrum-csv "$BASE/group_safe_1to10/data/noisy_subbin_spectrum_long.csv" \
  --split-csv "$BASE/group_safe_1to10/data/split_assignments.csv"
```

v4 调参必须只跑 train/val。依次运行 `(top-k, rssi-class-k)` 为
`(3,1)`、`(5,1)`、`(5,3)` 的三个配置，分别使用独立输出目录：

```bash
python3 -B "$BASE/run_aco_v4_on_split.py" \
  --splits train_loocv,val \
  --top-k 5 --rssi-class-k 3 \
  --rssi-csv "$BASE/group_safe_1to10/data/noisy_rssi_plus_packet_level_54points.csv" \
  --spectrum-csv "$BASE/group_safe_1to10/data/noisy_subbin_spectrum_long.csv" \
  --split-csv "$BASE/group_safe_1to10/data/split_assignments.csv" \
  --aco-v2-dir "$BASE/group_safe_1to10/results/aco_v2" \
  --result-dir "$BASE/group_safe_1to10/results_tuning" \
  --output-dir "$BASE/group_safe_1to10/results_tuning/aco_v4_top5_k3"
```

val 锁定 Top5-k3 后，最终 refit 只触碰 test：

```bash
python3 -B "$BASE/run_aco_v4_on_split.py" \
  --splits test \
  --top-k 5 --rssi-class-k 3 \
  --rssi-csv "$BASE/group_safe_trainval_refit/data/noisy_rssi_plus_packet_level_54points.csv" \
  --spectrum-csv "$BASE/group_safe_trainval_refit/data/noisy_subbin_spectrum_long.csv" \
  --split-csv "$BASE/group_safe_trainval_refit/data/split_assignments.csv" \
  --aco-v2-dir "$BASE/group_safe_1to10/results/aco_v2" \
  --result-dir "$BASE/group_safe_trainval_refit/results" \
  --output-dir "$BASE/group_safe_trainval_refit/results/aco_v4_top5_k3_refit"
```

LoRa-only 消融使用同一组参数，把脚本替换为
`run_aco_v4_lora_only_ablation.py`，输出到独立目录。

## 6. 搭档接续对比实验优先级

- [ ] 在最终 refit/test 协议下重跑 1-NN、KNN、PGAR，补齐传统基线统一表。
- [ ] 在最终 refit/test 协议下重跑 ACO v2；当前 63/74 来自 refit 前 source-safe split。
- [ ] 冻结参数后对 v4 做 no-Score4、no-prior、no-stability、no-reliability 分量消融。
- [ ] 若保留 v4.7 对比，按同一 refit/test 协议重跑；当前没有 v4.7 refit 结果。
- [ ] 测最终 74 包主线的端到端 latency 和 peak memory，明确是否包含模板构建。
- [ ] 报告多 seed 或置信区间；禁止用 test 选择 seed、阈值或规则。

每个新实验目录至少保存：命令、Git commit、seed、输入 SHA-256、协议标签、
`summary.csv/json` 和 test predictions。大体积 candidate/segment 中间表本地保留即可。

## 7. 禁用和历史内容

- `v2_output_wrong/`：明确错误数据，不得用于任何结果。
- `fingerprint_localization/model/v2/rerun_20260622/`：6/23 主线前的旧重跑，不纳入论文主线。
- `fingerprint_localization/experiments/aco_source_safe_1to10/data` 与 `results`：先增强再切分的泄漏协议，只能标为 historical/deprecated。
- v4.2-v4.8：除非在冻结的 source-safe/refit 协议上重跑，否则只作为探索记录。
- 7/10-7/11 chirp-LoRa 直接匹配实验：是重要负结果，用于限制论文表述，不应删除。

## 8. 当前仍不能完全从原始 IQ 复现的部分

上游 LoRa 特征提取器 `../gr-lora_sdr/examples/lora_file_preamble_fft.py` 不在本仓库。
因此 GitHub 版本应把 6/23 的预计算 RSSI/raw 输入和 6/24 的 q1/q4 频谱长表
作为正式公开输入，并记录校验和。原始 IQ 到这些特征的端到端复现，需后续补回
提取器、版本、参数和数据许可。
