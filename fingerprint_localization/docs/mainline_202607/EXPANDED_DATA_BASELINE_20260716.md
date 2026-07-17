# ExpandedReal-649-v1 数据基线锁定

更新日期：2026-07-16

## 锁定结论

后续数据预处理、数据增强、训练/验证/测试划分和算法对比，默认以
`ExpandedReal-649-v1` 为原始包基线。不再以 370 包旧 dense-detection 数据作为默认
起点。

## 口径

- 地图：现有 32 个位置。
- 真实 RSSI 对应物理槽：661。
- 检测 IQ 候选：651。
- 最终通过 QC 的真实包：649。
- 保留的旧基线物理 starts：364。
- 新恢复或重新对齐的包：285。
- 相对 370 包历史基线净增：279（+75.41%）。
- 位置覆盖：32/32，每位置 19--21 包。
- 当前版本是合并全集，不含 train/val/test 划分。

## 已锁定派生版本（2026-07-16）

算法实验默认使用 `ExpandedReal-649-v1-source-safe-1to10-v1`：

- 随机种子：`20260626`。
- source-first 分层划分：train/val/test = `393/128/128`，三组 source overlap 均为 0。
- 32 个位置在 train、val、test 中均有覆盖。
- 仅训练 source 做 1:10 增强；算法 packet rows = `3930/128/128`，总计 4186。
- RSSI 与 S17 噪声标准差为 `train source column std / 10`，不读取 val/test 统计量。
- q4 在 dB 域做平滑高斯扰动并重新归一化；训练增强副本的 q1 整数 bin 与 q4 对应点强制一致，幅度、dB、实部和虚部联动重算。
- val/test 的 256 个原包逐字段保持父数据不变，不做增强或数值修补。
- RSSI、RSSI+S17、PGAR 同时输出 raw 和 train-only standardized 矩阵。
- 独立验证结果：`PASS`；PGAR 与 ACO v2 原生 loader 均成功读取 4186 个 packet。

派生数据目录：

`data/expanded_real_32points_20260716/source_safe_1to10/ExpandedReal649_source_safe_1to10_seed20260626/`

其中 `data/split_assignments.csv` 和 NPZ 内的 `train_indices/val_indices/test_indices`
是后续实验的唯一划分口径。旧主线 74 包 test 成绩仅保留为历史结果，不得与本版本
128 包 test 直接比较。

## Expanded ACO v4 主线结果（2026-07-16）

冻结旧主线配置后在 Expanded split 上执行：

- 配置：ACO v4，`top_k=5`，`rssi_class_k=3`，4 segments，seed `20260626`，
  `T_seg=0.02403918092026683`。
- train-only validation：`94/128 = 73.44%`。
- train-only test 诊断：`90/128 = 70.31%`。
- 正式 train+validation 1:10 refit：521 个 source、5210 个训练 rows。
- 正式 test：`98/128 = 76.56%`，95% Wilson CI 为 `68.52%--83.06%`。
- 同 test RSSI Top-1：`84/128 = 65.63%`；RSSI Top-5 recall：`119/128 = 92.97%`。
- ACO 相对 RSSI 为 18 W2R、4 R2W，净增 14，McNemar exact 双侧 `p=0.00434`。
- 正式 refit 相对 train-only test 改变 17 个预测：9 W2R、1 R2W、7 个改判后仍错误。
- Score4 相对 vote 改判 0 个，本轮不能声称 Score4 带来准确率增益。

正式 refit 继续只使用原 393 个 train source 估计的噪声统计；原 val source 在配置冻结后
才生成训练副本。128 个 test 包在 RSSI、S17、频谱中逐字段等同父数据，test 与 refit train
source overlap 为 0。完整验证结果为 `PASS`。

结果目录：`results/expanded_source_safe_1to10/`。

正式 refit 数据目录：
`data/expanded_real_32points_20260716/trainval_refit/ExpandedReal649_trainval_refit_seed20260626/`。

详细说明：`docs/mainline_202607/EXPANDED_ACO_MAINLINE_20260716.md`。

## 协议

1. source ID 为 `(file_stem, packet_index)`。
2. packet index 表示 5 s 物理周期槽，不是漏检后的连续 dense index。
3. 每个 IQ 文件使用可信 starts 稳健估计实际周期和相位。
4. 每包使用 16 个 preamble symbols，q=1/q=4，per-symbol alignment，peak normalization。
5. 最终 RSSI、packet summary 和 spectrum long 的 source key 完全一致。
6. 后续若划分数据，必须先按 source packet 划分，再只对训练 source 做增强。

## 源文件

- `data/expanded_real_32points_20260716/final/rssi_plus_packet_level_32points_expanded.csv`
- `data/expanded_real_32points_20260716/final/subbin_spectrum_long_32points_expanded.csv`
- `data/expanded_real_32points_20260716/final/accepted_packet_starts.csv`
- `data/expanded_real_32points_20260716/final/source_packet_manifest.csv`
- `data/expanded_real_32points_20260716/final/EXPANDED_DATASET_REPORT.json`

## 旧数据边界

旧 370 包输入保留不变，仅作历史对照。6 个旧 starts 超出对应 RSSI 物理槽范围，
已记录在 `legacy_unpaired_packet_starts.csv`，不得通过重用 dense packet index 强行配对。
