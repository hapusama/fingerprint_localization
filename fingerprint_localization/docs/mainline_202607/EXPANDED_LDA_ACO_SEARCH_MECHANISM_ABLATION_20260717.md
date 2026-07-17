# EXPANDED LDA/ACO 搜索机制消融（2026-07-17）

## 目的

在 `EXPANDED_LDA_ACO_MAINLINE_20260716` 上隔离“信息素”和“路径搜索”的实际贡献。四种方法使用完全相同的 LDA 后验、RSSI/LDA 融合 Top-5 候选、segment 观测代价、RSSI/raw prior 和最终 `beta=0.5` LDA 融合，只更换候选内部搜索/评分机制。

## 受控变体

| 变体 | 保留内容 | 移除内容 |
| --- | --- | --- |
| 平均代价 | 每个候选的平均 `C_obs` | 路径、转移代价、蚂蚁、信息素 |
| 贪心路径 | 动态 switch penalty，每步选当前最小代价 | 多路径搜索、信息素 |
| 无信息素 | 16 ants × 12 iterations、启发式转移、路径代价、elite vote | `tau`、挥发、增强、pheromone score |
| 完整 ACO | 冻结的 ACO v4 + Score4 | 无 |

无信息素和完整 ACO 使用两个独立 RNG，但以相同 seed 初始化，以避免人为把随机性偏向某一方。

## 数据和完整性

- 干净验证集：128 包。
- Expanded formal test：128 包；干净场景加 14 个人工退化场景，共 15 个 packet-scenario 组。
- 人工退化：前导码缺失 1/2/4 个符号，幅度噪声 0.05/0.1/0.25/0.5/1.0，CFO 0.25/0.5/1.0 bin，segment 异常 0.25/0.5/1.0。
- 正式预测记录：7,680 行（128 包 × 15 场景 × 4 方法）。
- 完整 ACO 与前一次受控弱包实验逐包复现：0 个不一致。

formal test 已在早期实验中查看过，因此本轮 formal 结果是 exploratory，不是全新盲测证据。由于没有原始 IQ，人工退化是 feature-space proxy。

## 干净数据

| 方法 | 验证 search/final 正确数 | formal search/final 正确数 | formal final accuracy | MAE/P95 (m) | 严重错误率 | 平均搜索耗时/packet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 平均代价 | 117/118 | 114/121 | 94.53% | 0.289/2.204 | 0.78% | 0.001 ms |
| 贪心路径 | 117/116 | 117/121 | 94.53% | 0.314/2.204 | 0.78% | 0.079 ms |
| 无信息素 | 117/117 | 117/121 | 94.53% | 0.289/2.204 | 0.78% | 4.68 ms |
| 完整 ACO | 117/117 | 118/120 | 93.75% | 0.366/3.390 | 0.78% | 4.90 ms |

完整 ACO 在 formal 的纯搜索 Top-1 上多对 1 包，但在统一的 LDA 最终融合后少对 1 包；验证集上完整 ACO 与无信息素完全持平。

## 15 个 formal 场景的胜/平/负

以 final accuracy 的逐场景正确包数计算：

| 完整 ACO 相对于 | 胜/平/负 | 15 场景累计净正确包数 |
| --- | ---: | ---: |
| 平均代价 | 7/2/6 | +9 |
| 贪心路径 | 7/2/6 | +4 |
| 无信息素 | 4/3/8 | -8 |

完整 ACO 能优于极简平均代价和贪心路径，但无信息素变体在更多场景中胜出，且累计多对 8 个 packet-scenario。

## 完整 ACO 的局部优势

| 场景 | 无信息素 final | 完整 ACO final | 正确包差 | 无信息素 MAE | 完整 ACO MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 幅度噪声 0.25 | 57.03% | 60.94% | +5 | 2.750 m | 2.273 m |
| 幅度噪声 0.5 | 39.06% | 40.63% | +2 | 4.629 m | 4.549 m |
| 前导码缺失 4 | 81.25% | 82.81% | +2 | 0.760 m | 0.759 m |

幅度噪声 0.25 是完整 ACO 相对无信息素的最强单场景结果，但配对 McNemar 双侧 `p=0.0625`，仍未达到 0.05。所有场景的完整 ACO vs. 无信息素配对检验都未达 `p<0.05`。

## 已有诊断量的 severe 分组

| severe 分组 | 平均代价 | 贪心路径 | 无信息素 | 完整 ACO |
| --- | ---: | ---: | ---: | ---: |
| 低 detect score，n=31 | 83.87% | **87.10%** | 83.87% | 83.87% |
| 低 SNR，n=59 | 88.14% | **89.83%** | 88.14% | 88.14% |
| 高 segment cost std，n=41 | **95.12%** | 92.68% | **95.12%** | 92.68% |

完整 ACO 在三个 severe 诊断分组中均不是最优。

## 机制层面的关键发现

- 完整 ACO 和无信息素的 best path 在 15 个 formal 场景中的平均 segment switch 均为 0，garbage segment 也均为 0。
- 这意味着当前最优路径实际上是“所有 segment 选同一个位置”，多 segment 路径组合没有被利用。
- 完整 ACO 每包平均搜索约 4.90 ms，约为贪心路径的 62 倍；而完整 ACO 与无信息素的耗时接近。
- 因此，当前结果更像“带信息素的随朷重采样在特定噪声段有局部收益”，而不是“路径式 ACO 全局不可替代”。

## 结论

**这组消融目前不能证明蚂群搜索机制不可替代。**

可以被数据支持的较弱表述是：信息素增强在中等幅度噪声和大量前导码缺失下有局部优势，但这个优势尚未在配对显著性、干净验证、CFO、segment 异常或已有 severe 分组上形成一致证据。

如果论文需要强“不可替代”结论，下一轮必须首先让 path 产生可观测的非零 segment switch/异常拒绝，再使用多 seed、多 split/新盲测包和原始 IQ 扰动做配对置信区间。否则应将 ACO 定位为“特定强退化区间的可选鲁棒化模块”，而非通用主线的必要组件。

## 复现

```bash
PYTHONPATH=/tmp/fingerloc_py39_20260716:fingerprint_localization/experiments/aco_source_safe_1to10 \
  /usr/bin/python3 -B \
  fingerprint_localization/experiments/aco_source_safe_1to10/run_search_mechanism_ablation.py

/Users/siri/radioconda/bin/python3.12 -B \
  fingerprint_localization/experiments/aco_source_safe_1to10/plot_search_mechanism_ablation.py
```

完整结果位于 `fingerprint_localization/results/expanded_source_safe_1to10/search_mechanism_ablation_20260717/`。
