# ExpandedReal ACO v4 主线实验

更新日期：2026-07-16

## 正式结果

在 `ExpandedReal-649-v1-source-safe-trainval-refit-v1` 上，冻结的 ACO v4
`top_k=5, rssi_class_k=3` 最终结果为：

- test：`98/128 = 76.56%`。
- 95% Wilson CI：`68.52%--83.06%`。
- 同 test RSSI Top-1：`84/128 = 65.63%`。
- RSSI Top-5 recall：`119/128 = 92.97%`。
- ACO 相对 RSSI：18 W2R、4 R2W，净增 14 个正确包。
- McNemar exact 双侧检验：`p=0.00434`。

旧 `67/74 = 90.54%` 使用不同 source 集合和 test split，只能作为历史结果，
不得与 `98/128` 直接比较百分比。

## 协议

- 原始 source split：393/128/128，seed `20260626`。
- validation 配置评估只使用 3930 个 train 增强 rows，得到 `94/128 = 73.44%`。
- 配置冻结后，原 train+val 共 521 个 source 形成正式 refit train。
- 393 个原 train source 沿用已有 3930 个增强 rows。
- 128 个原 val source 使用只由原 train 估计的噪声统计生成 1280 个增强 rows。
- 正式 refit train 共 5210 rows；128 个 test 原包不增强。
- test source 与 refit train source overlap 为 0。
- 128 个 test 包在 RSSI、S17、频谱文件中逐字段等同父数据。

冻结参数：

- `top_k=5`
- `rssi_class_k=3`
- `segment_count=4`
- `seed=20260626`
- `T_seg=0.02403918092026683`
- 其余 ACO v4 参数使用主线默认值。

## 诊断

| 运行 | 训练口径 | test/val | 正确率 |
| --- | --- | ---: | ---: |
| train-only validation | 3930 train rows | 94/128 | 73.44% |
| train-only test | 3930 train rows | 90/128 | 70.31% |
| 直接 refit 诊断 | 3930 train rows + 128 val originals | 92/128 | 71.88% |
| 正式 1:10 refit | 5210 train+val rows | **98/128** | **76.56%** |

正式 refit 相对 train-only test 改变 17 个预测，其中 9 个 W2R、1 个 R2W、
7 个改判后仍错误，净增 8 个正确包。

最终仍有 30 个错误：9 个真实位置未进入 RSSI Top-5，21 个真实位置已在候选集内但
ACO 仍选择错误。最弱位置为 `1_23`（0/4），其次是 `0_42` 和 `1_21`（各 1/4）。
最大重复混淆为 `1_21 -> 0_42`（3 包）。

本轮 Score4 相对 ACO vote 的改判数为 0，因此本次准确率提升来自 ACO 路径搜索相对
RSSI 的纠错，不能归因于 Score4 后融合。

## 文件

- 正式预测：`results/expanded_source_safe_1to10/aco_v4_top5_k3_formal_1to10_refit/test_predictions.csv`
- 正式指标：`results/expanded_source_safe_1to10/aco_v4_top5_k3_formal_1to10_refit/aco_v4_split_metrics.json`
- 汇总报告：`results/expanded_source_safe_1to10/report/expanded_aco_v4_mainline_report.json`
- 逐位置结果：`results/expanded_source_safe_1to10/report/formal_refit_per_location.csv`
- 混淆对：`results/expanded_source_safe_1to10/report/formal_refit_confusion_pairs.csv`
- 运行清单与哈希：`results/expanded_source_safe_1to10/report/RUN_MANIFEST.json`
- refit 数据验证：`data/expanded_real_32points_20260716/trainval_refit/ExpandedReal649_trainval_refit_seed20260626/metadata/validation_report.json`

## Source-level 挑战实验

后续验证了 source-level 模板和 `T_seg`：

- 只将模板、方差、可靠性和 shrinkage 改为 source-level，并用 5-fold source OOF 标定
  `T_seg`：validation 仍为 `94/128`，formal test 为 `96/128`，低于主线 2 包。
- 在上述基础上，再将 RSSI Top-5 近邻也按 source 聚合：validation `99/128`，formal test
  `102/128 = 79.69%`。
- 完整 source-level 版本相对 `98/128` 主线为 8 W2R、4 R2W，净增 4；McNemar exact
  双侧 `p=0.388`，当前证据不足以替换主线。
- source-level formal refit 的 `T_seg=0.0090341267`，显著低于旧冻结值 `0.0240391809`。
- shrinkage `alpha` 从 row-level 平均约 `0.953` 降为 source-level 平均约 `0.670`。

详细报告：`docs/mainline_202607/SOURCE_LEVEL_ACO_20260716.md`。
