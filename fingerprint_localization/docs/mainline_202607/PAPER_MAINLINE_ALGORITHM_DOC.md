# Mainline Algorithm Document

本文主线方法采用宽带 chirp 辅助的 LoRa 指纹定位框架。宽带 chirp 只在离线阶段参与模板构建，用于提供位置级多径先验和模板可靠性调权；在线定位阶段只使用 LoRa/RSSI/q=4 频谱观测。

## 1. 实验协议

为避免数据泄露，实验采用 source-packet-safe 划分方式。

1. 先按原始 source packet 划分 train、validation、test。
2. train 中的 source packet 才生成 1:10 增强样本。
3. validation 和 test 均保持原始未增强 packet。
4. 只使用旧独立 validation 选择配置。
5. 配置固定后，将 train+validation 的 source packet 合并为最终训练源，再做 1:10 增强。
6. test 保持原始 74 个 source packet，不增强、不参与调参。
7. final refit 中 test source overlap with train 为 0。

最终配置由旧独立 validation 选出：

| item | value |
|---|---:|
| Method | ACO v4 |
| `top_k` | 5 |
| `rssi_class_k` | 3 |
| `segment_count` | 4 |
| `ants` | 16 |
| `iterations` | 12 |
| `elite_ants` | 4 |

最终报告使用 `ACO v4 top5 k3 + trainval refit`：

| split | packets | correct | accuracy |
|---|---:|---:|---:|
| test | 74 | 67 | 0.9054 |

## 2. 输入数据

每个 LoRa packet 包含两类在线观测：

1. RSSI/packet-level features。
2. q=4 zero-padding FFT 后的 LoRa 频谱形态。

每个位置还包含离线宽带 chirp 信息：

1. chirp-derived raw-bin prior shape。
2. chirp multipath structure，包括多径复杂度、`k_ratio`、`tau_rms` 等结构参数。

在线测试时不需要采集 chirp；chirp 只用于离线模板构建。

## 3. 离线模板构建

对每个位置 \(i\)，首先从训练 LoRa packet 中构建经验模板：

\[
\mu_i^{emp}, \quad \sigma_i^{2,emp}
\]

其中经验模板来自 LoRa raw-bin/q=4 分段频谱特征。

同时，从宽带 chirp 得到物理先验模板：

\[
\mu_i^{phy}
\]

以及结构参数：

\[
\text{k\_ratio}_i,\quad \tau_{rms,i}
\]

如果某个位置没有实测 chirp，则使用空间插值或全局 fallback。代码中通过 `chirp_source` 记录来源，例如 `measured_chirp`、`interpolated`、`nearest`、`empirical_global_fallback`。

### 3.1 模板均值收缩

位置 \(i\) 的最终 raw-bin 模板均值由 LoRa 经验模板和 chirp 先验模板融合得到：

\[
\mu_i = \alpha_i \mu_i^{emp} + (1-\alpha_i)\mu_i^{phy}
\]

其中：

\[
\alpha_i = \frac{n_i}{n_i+\lambda}
\]

\(n_i\) 是该位置训练 packet 数量，\(\lambda\) 是 shrinkage 参数。

直观解释：训练样本足够多时，模板更相信 LoRa 经验统计；训练样本较少或不稳定时，模板更多借助 chirp 多径先验。

### 3.2 模板方差构建

chirp 结构参数用于构建物理方差：

\[
\sigma_i^{2,phy}
= c_0 + \frac{c_1}{\text{k\_ratio}_i+1} + c_2 \tau_{rms,i}
\]

最终模板方差为：

\[
\sigma_i^2
= \alpha_i \sigma_i^{2,emp}
+ (1-\alpha_i)\sigma_i^{2,phy}
+ \sigma_0^2
\]

模板可靠性定义为：

\[
r_i = \frac{1}{\sum_j \sigma_{i,j}^2+\epsilon}
\]

该可靠性后续进入 ACO 的 self-pheromone 和候选评分。

## 4. 在线候选生成

对测试 packet \(x\)，首先使用 RSSI 分类距离得到候选位置排序。

取前 `top_k=5` 个位置作为 ACO 搜索候选集合：

\[
\mathcal{C}(x)=\{c_1,c_2,\ldots,c_5\}
\]

RSSI top-k 只负责保证候选召回，最终选择由 ACO 完成。

## 5. LoRa 分段观测代价

每个 LoRa packet 被划分为 `segment_count=4` 个 segment。

对每个 segment \(s\) 和候选位置 \(i\)，计算多项观测代价：

| symbol | meaning |
|---|---|
| \(C_R\) | RSSI candidate cost |
| \(C_{bin}\) | raw-bin template cost |
| \(C_E\) | energy prototype cost |
| \(C_W\) | LoRa spectrum/vector prototype cost |
| \(C_Q\) | q=4 shape cost，只有 q=4 segment 可靠时启用 |

基础 LoRa segment cost 为：

\[
C_{seg}
= w_E C_E + w_W C_W + w_Q C_Q + w_B C_{bin}
\]

当前主线中：

| weight | value |
|---|---:|
| `energy_weight` | 0.20 |
| `raw_weight` | 0.55 |
| `bin_weight` | 0.02 |
| `q4_weight` | 0.00 |
| `rssi_weight` | 0.45 |

最终 segment observation cost 为：

\[
C_{obs}=w_R C_R + Q_{seg} C_{seg}
\]

其中 \(Q_{seg}\) 描述 packet 内 segment cost 的稳定性：

\[
Q_{seg}=\frac{1}{1+\frac{\text{std}(C_{seg}^{min})}{T_{seg}}}
\]

当 segment 间代价不稳定时，\(Q_{seg}\) 变小，算法会降低 LoRa segment evidence 的影响。

## 6. ACO 路径搜索

ACO 将一个 packet 的多个 segment 看作一条路径。每个 segment 需要选择一个候选位置，或选择 garbage state。

路径代价包括：

1. segment observation cost；
2. RSSI top1 prior；
3. raw-bin winner prior；
4. cost veto；
5. 候选之间切换惩罚；
6. garbage state 使用惩罚；
7. 多候选分散惩罚。

路径总代价可概括为：

\[
J(P)=
\sum_s C_{obs}(s,P_s)
- \sum_s \text{Prior}(P_s)
- \sum_s \log V(P_s)
+ \sum_s \text{SwitchPenalty}(P_{s-1},P_s)
+ \lambda_G N_G
+ \lambda_D N_U
\]

其中 \(N_G\) 是 garbage state 数量，\(N_U\) 是路径中使用过的非 garbage 候选数量。

v4 中切换惩罚受 segment 稳定性调节：

\[
\text{SwitchPenalty}_{v4}
= \text{SwitchPenalty}_{v2}
\cdot (1+\lambda_Q(1-Q_{seg}))
\]

当 packet 内 segment evidence 不稳定时，算法更不鼓励频繁切换候选。

## 7. 信息素与可靠性

ACO 初始化和更新 self-pheromone 时使用模板可靠性：

\[
\tau_{ii}
= \tau_{stay}
\left(1+\lambda_R I[i=\text{RSSI top1}]
+ \lambda_C \frac{r_i}{\text{median}(r)}
\right)
\]

其中 \(r_i\) 来自离线模板方差，因此 chirp 通过模板可靠性间接影响 ACO 搜索。

elite ants 会根据低代价路径更新信息素，并统计每个候选的 elite vote。

## 8. Score4 最终选择器

ACO v4 不只使用单一路径结果，而是综合：

1. self-pheromone；
2. elite vote；
3. normalized observation cost；
4. RSSI top1 prior；
5. raw-bin winner prior。

最终候选得分为：

\[
S_i
= z(\tau_{ii})
+ \lambda_{vote} z(V_i)
- \lambda_{cost} z(C_i)
+ \lambda_R I[i=\text{RSSI top1}]
+ \lambda_W M_{raw}I[i=\text{raw winner}]
\]

最终输出：

\[
\hat y=\arg\max_i S_i
\]

## 9. ACO v2.0、v4.0、v4.7 的角色

### ACO v2.0

v2.0 是基础 ACO 框架，包含：

1. RSSI top-k 候选；
2. LoRa segment path search；
3. chirp-shrinkage templates；
4. garbage state；
5. dynamic switch penalty。

### ACO v4.0

v4.0 是论文主方法，继承 v2.0 并加入：

1. RSSI top1 weak prior；
2. raw-bin winner weak prior；
3. segment-stability gating \(Q_{seg}\)；
4. cost veto；
5. reliability-weighted self-pheromone；
6. Score4 final selector。

论文最终结果使用 ACO v4 top5 k3 + trainval refit。

### ACO v4.7

v4.7 是 two-stage rescue/rule branch，用于比较和验证。当前 7.10 final refit 中没有 v4.7 refit 输出，因此论文中不应把 v4.7 描述为最终主结果。

## 10. Chirp 消融结论

已测试 LoRa-only 离线模板消融，即关闭 chirp field，仅用 LoRa 数据构建模板。

| split | with chirp | LoRa-only | change |
|---|---:|---:|---:|
| train_loocv | 2697/2960 | 2698/2960 | +1 |
| val | 70/73 | 70/73 | 0 |
| test | 67/74 | 66/74 | -1 |

结论：chirp 不是在线定位输入，也不是决定性单独特征，但作为离线模板先验带来小幅正收益。在 test 中，chirp 帮助修正了一个 LoRa-only 会误判的 packet。

## 11. 推荐论文表述

可以将方法描述为：

> We propose a source-safe, chirp-assisted ACO fingerprinting method for LoRa localization. Wideband chirp measurements are used only during offline template construction to regularize LoRa spectral templates and estimate template reliability. During online inference, the system uses only LoRa packet observations. RSSI first provides a top-k candidate set, and an evidence-reliability-aware ACO searches over segment-level candidate paths. The final location is selected by a Score4 fusion of self-pheromone, elite votes, observation cost, and weak RSSI/raw-bin priors.

中文表述：

> 本文提出一种宽带 chirp 辅助的 LoRa 蚁群指纹定位方法。宽带 chirp 仅在离线阶段用于构建位置级多径先验和模板可靠性，不作为在线输入。在线定位时，系统首先利用 RSSI 生成候选位置集合，然后基于 LoRa 分段频谱观测构建路径代价，并通过可靠性感知的蚁群搜索选择最一致的位置。最终结果由信息素、elite vote、观测代价以及弱 RSSI/raw-bin 先验共同决定。

## 12. Key Files

Mainline lock:

- `MAINLINE_20260710_LOCK.md`

Source-safe data protocol:

- `build_group_safe_1to10_data.py`
- `group_safe_1to10/data/group_safe_metadata.json`
- `group_safe_trainval_refit/data/refit_metadata.json`

Configuration selection:

- `group_safe_1to10/results_tuning/aco_v4_top5_k3/aco_v4_summary.csv`

Final refit result:

- `group_safe_trainval_refit/results/aco_v4_top5_k3_refit/aco_v4_summary.csv`
- `group_safe_trainval_refit/results/method_summary_with_aco_v4.csv`
- `group_safe_trainval_refit/results/refit_comparison_summary.json`

Implementation:

- `run_aco_v4_on_split.py`
- `fingerprint_localization/model/v3/aco_packet_path_v2.py`
- `fingerprint_localization/model/v3/aco_packet_path_v4.py`

LoRa-only ablation:

- `run_aco_v4_lora_only_ablation.py`
- `group_safe_trainval_refit/results/aco_v4_top5_k3_refit_lora_only/lora_only_vs_mainline_summary.csv`
