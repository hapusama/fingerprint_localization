# Expanded-649 ACO 学习候选先验实验

> **历史版本，已被替代。** 当前主线已切换为不含RF的LDA-only full ACO，validation
> 117/128、formal 120/128。请使用
> `docs/mainline_202607/EXPANDED_LDA_ACO_MAINLINE_20260716.md`。本文件保留LDA+RF(700)
> 119/128的实验过程与对照证据。

更新日期：2026-07-16

## 结论

在 `ExpandedReal-649-v1` 固定 source-safe 协议上，ACO v4 加入离线学习候选先验和
验证集约束的 Score4 概率融合后，正式 test 达到：

- Validation：116/128 = 90.63%。
- Formal test：119/128 = 92.97%。
- 95% Wilson CI：87.18%--96.26%。
- 对应旧 90.54% 的 Expanded 门槛为 116/128，当前超过 3 个包。

该结果满足 Expanded 数据集上保持原准确率水平的目标，但属于探索性结果，因为同一
Expanded test 在此前实验中已经多次查看。

## 协议

- Dataset：`ExpandedReal-649-v1`。
- Source split seed：`20260626`。
- Validation 阶段：393 个 train source 做 1:10，128 个 validation 原包不增强。
- Formal refit：521 个 train+validation source，5210 个增强训练行。
- Formal test：128 个原始真实包，不增强。
- Train/validation/test source overlap：0。
- ACO：Top-5、RSSI class k=3、4 segments。

## 方法

学习候选先验由标准化 LDA(svd) 与 RF(700 trees, min leaf=2) 的类别后验等权组成。
该后验不是直接替代 ACO，而是先进入 RSSI 候选排序和先验代价：

```text
C_candidate = (1-alpha) * C_RSSI + alpha * scaled(-log P_LDA+RF)
```

Validation 扫描 `alpha=0.0--1.0`，同分时选择更小学习权重。最优为 `alpha=0.3`：

- 无学习先验：99/128。
- `alpha=0.3` 完整 ACO：115/128。

最终选择仍限制在 ACO 产生的候选集内。为在达到旧准确率目标的同时尽量保持 ACO
占比，validation 使用“达到 116/128 的最小 beta”规则：

```text
Score_final = (1-beta) * norm(Score4_ACO) + beta * norm(P_LDA+RF)
```

Validation 选择 `beta=0.5`，即 ACO Score4 与学习先验各占 50%。

`700 trees` 是本轮监督基线和候选先验采用的固定高容量 RF 配置，不是通过独立的
`n_estimators` validation sweep 选出的最优值。因此论文应将其写成固定实现参数，不能
声称 700 是最优树数。若后续减少树数，必须只在 validation 上选择树数，再冻结后评估，
不能根据当前已查看的 formal test 选择。

当前主线决策为继续冻结 **700 trees**。100 树 validation-only 压缩在 formal test 降为
118/128；150 树虽然在 post-hoc test audit 中保持 119/128，但改变了 3 个最终标签，不能
替换当前冻结模型。树数敏感度只保留为工程诊断。

## 正式结果

| 阶段 | Correct | Accuracy |
|---|---:|---:|
| Source-level ACO，无学习先验 | 102/128 | 79.69% |
| 学习先验进入候选生成和 ACO | 114/128 | 89.06% |
| ACO Score4 + posterior，beta=0.5 | 119/128 | 92.97% |

最终版本相对无先验 source-level ACO：

- 19 W2R、2 R2W，净增 17。
- McNemar exact 双侧 `p=0.000221`。

相对先前候选排序版本 105/128：

- 16 W2R、2 R2W，净增 14。
- McNemar exact 双侧 `p=0.001312`。

最终 Score4 posterior 相对尚未融合 posterior 的 114/128：5 W2R、0 R2W，
McNemar `p=0.0625`。

## 纯 LDA+RF 对照与 ACO 价值

同一 700 树 RF、同一 formal refit/test 上，纯 LDA+RF posterior Top-1 为：

```text
120/128 = 93.75%
```

当前 ACO candidate-prior/Score4 融合为 119/128，因此 ACO 在这个 split 上没有带来净
accuracy。两者有 9 个最终分歧：ACO 修正 4 个 ML 错误，同时破坏 5 个 ML 正确结果，
净 -1；没有“错误改为另一错误”的分歧。5 个 R2W 中有 2 个是 ML 真值 Top-1 未进入
ACO Top-5，最终候选约束强制选择了错误位置。

| 方法 | Correct | Feature-ready Median | Packet-to-location Median | P95 |
|---|---:|---:|---:|---:|
| 纯 LDA+RF(700) | 120/128 | 13.342 ms | 14.662 ms | 15.272 ms |
| ACO + learned prior + final fusion | 119/128 | 20.691 ms | 28.903 ms | 29.746 ms |

ACO 当前已经实现并可直接审计的非 accuracy 价值包括：

1. 每个包输出 4 段 candidate observation costs，可分解为 RSSI、raw-bin、energy/vector、
   stability、veto 和 template reliability，而不是只给黑盒 posterior。
2. 保存 best path、garbage state、self-pheromone、elite vote 和 Score4，可定位某次改判
   来自候选排除、段间不一致、路径搜索还是 final fusion。
3. 最终位置受 Top-5 和段级路径一致性约束，形成明确的物理证据边界；9 个分歧中的
   4 W2R 证明物理路径证据与 ML posterior 存在互补性。
4. learned posterior、RSSI 候选和物理观测是模块化接口，未来可在不重写 ACO 路径层的
   前提下替换分类器或重新标定物理模板。

尚不能作为已证实贡献声称的内容包括：ACO 比纯 ML 更准确、在新环境更鲁棒、具有更好
校准或 OOD 检测能力。要把这些作为论文优势，仍需独立环境、受控噪声/弱包、候选召回和
selective-risk 实验。当前最严谨的叙事是：ACO 用约 14.2 ms 额外 packet-to-location 计算
换取候选约束、段级物理证据融合和逐包可解释诊断，但在当前 formal split 上 accuracy
比纯 LDA+RF 少 1 包。

### LDA/RF 组件消融（validation only）

固定 `alpha=0.3`、`beta=0.5`、`T_seg=0.009161130588433`、RF=700 和全部 ACO 参数，
只改变 posterior 组件；该实验没有加载 formal test：

| Posterior | ML Top-1 | Candidate-rank Top-1 | True Top-5 recall | ACO Score4 | Final |
|---|---:|---:|---:|---:|---:|
| LDA-only | 117/128 | **117/128** | **127/128** | **117/128** | **117/128** |
| RF-only | 113/128 | 110/128 | 124/128 | 111/128 | 112/128 |
| LDA+RF equal | **119/128** | 112/128 | 125/128 | 115/128 | 116/128 |

结论分为两层：等权组合对纯 posterior Top-1 有 validation 依据，相对 LDA-only 为
2 W2R、0 R2W；但在固定完整 ACO 管线中，LDA-only final 反而比组合多 1 包。组合相对
LDA-only 改变 4 个最终标签，1 W2R、2 R2W，净 -1。RF posterior 改变候选排序后，真值
Top-5 recall 从 127/128 降到 125/128，是组合最终结果下降的重要边界。

因此不能再声称“LDA+RF 是完整 ACO validation 的最优组件组合”或“RF 对最终 ACO 必不可少”。
当前700树主线仍按既定决策冻结，不因这项 validation 消融直接切换到 LDA-only；若要改变
主线，应预先冻结 LDA-only 配置并使用新的独立 test 确认。

在用户明确询问后，对上述已冻结的 LDA-only 配置进行了独立目录的 formal audit。固定
`alpha=0.3`、`beta=0.5`、formal `T_seg=0.009034126697630788` 和全部 ACO 参数：

| LDA-only formal 阶段 | Correct |
|---|---:|
| Pure LDA posterior Top-1 | 120/128 |
| Candidate-rank Top-1 | 118/128 |
| True Top-5 recall | 128/128 |
| ACO Score4 | 118/128 |
| Fixed-beta final | **120/128 = 93.75%** |

相对冻结 LDA+RF final 119/128，LDA-only 改变 8 个最终标签，4 W2R、3 R2W、1 个错误
改为另一错误，净 +1。相对 pure LDA Top-1，ACO final 改变 4 个标签，2 W2R、2 R2W，
净增益为 0。该结果进一步说明 RF 与 ACO 在当前 split 上都没有提高 LDA-only 的最终
accuracy。

该 audit 不能视为新的 confirmatory test：Expanded formal test 在此前已多次查看，而且
LDA-only 是在 validation 组件消融之后才运行。当前700树主线仍按用户决策保持不变；
`120/128` 只记录为探索性候选结果。

## 在线定位时延

在 Apple M1 Pro（Python 3.8.9、NumPy 1.24.4、scikit-learn 1.3.2）上对冻结的
`alpha=0.3`、`beta=0.5` 最终方法重新计时。每轮均按固定顺序处理 128 个 formal-test
包，并强制与 `final_test_predictions.csv` 逐包核验：119/128 正确，0 个预测不一致。

| 计时边界 | Timed packets | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| RSSI+/S17/q1/q4 特征已在内存 -> 位置 | 1280 | 58.295 ms | 57.773 ms | 60.041 ms | 70.776 ms |
| 完整 RSSI/IQ 包在内存且包起点已知 -> 位置 | 640 | 66.031 ms | 65.706 ms | 67.590 ms | 68.409 ms |

Packet-to-location 口径包含 RSSI+、q1/q4、S17 特征提取，单包 LDA/RF posterior，
source-level 候选排序，四段观测代价，ACO 搜索/Score4 和最终 beta 融合；不包含空口接收、
包同步/起点检测、模型与模板预载和磁盘输出。原始特征现场重建核验中，q1/q4/S17
最大误差为 0，RSSI 最大浮点误差为 `2.84e-14`。

Feature-ready 中位数的主要组成是：LDA+RF posterior 39.544 ms、source-level 候选排序
11.968 ms、ACO 搜索与 Score4 4.905 ms、观测代价 1.071 ms。Packet-to-location 额外的
原始特征提取中位数为 7.933 ms。这里报告的是当前 Python 单包参考实现时延，不是经过
模型压缩、RF 串行化优化或候选统计缓存后的嵌入式下界。

### 保持预测不变的在线优化

在线单包推理无需修改或裁剪 700 棵树：将已加载 RF 的 `n_jobs` 从 `-1` 改为 `1`，避免
Joblib 为单行样本分发 700 棵树时产生的线程调度开销；同时缓存冻结训练集的 521 个
source-level RSSI 聚合向量及其均值/标准差。训练仍可使用 `n_jobs=-1`，该设置只作用于
部署阶段的单包 `predict_proba`。

| 实现 | Feature-ready Median | Feature-ready P95 | Packet-to-location Median | Packet-to-location P95 | Correct | Frozen mismatch |
|---|---:|---:|---:|---:|---:|---:|
| 原 Python 参考实现 | 57.773 ms | 60.041 ms | 65.706 ms | 67.590 ms | 119/128 | 0 |
| 单线程 RF + source 统计缓存 | 20.691 ms | 21.572 ms | 28.903 ms | 29.746 ms | 119/128 | 0 |

优化后 feature-ready 中位时延下降 64.2%，完整包到位置中位时延下降 56.0%。RF 单包
posterior 中位数由 39.252 ms 降至 12.809 ms，候选排序由 11.968 ms 降至 1.378 ms。
优化前后 RF posterior 的最大数值差为 `2.22e-16`，属于浮点求和顺序导致的机器精度
差异；候选集、ACO 路径输出和最终 128 个标签均保持不变。

## ACO 主体边界

最终系统仍包含并执行：

1. RSSI/学习先验候选生成；
2. 四分段物理观测代价；
3. 信息素迭代；
4. 精英路径投票；
5. ACO Score4；
6. 候选集内的等权 Score4/posterior 融合。

但必须如实记录：LDA+RF posterior 单独在同一 formal test 上为 120/128，混合 ACO 为
119/128。因此不能宣称 ACO 在该 split 上击败了所有监督分类器。ACO 论文叙事应聚焦
物理证据融合、候选约束、路径可解释性和弱先验下相对无先验 ACO 的显著提升，而不是
声称其纯 accuracy 高于 LDA+RF。

## 文件

- 旧 370 到 Expanded-649 变化说明：`docs/mainline_202607/OLD370_TO_EXPANDED649_MAINLINE_CHANGES_20260716.md`
- 当前主线核心代码文档：`docs/mainline_202607/EXPANDED649_CURRENT_MAINLINE_CORE_CODE_20260716.md`
- RF 树数敏感度与时延：`docs/mainline_202607/EXPANDED_RF_TREE_SENSITIVITY_20260716.md`
- 纯 LDA+RF 延时脚本：`experiments/aco_source_safe_1to10/benchmark_expanded_lda_rf_latency.py`
- LDA/RF 组件消融脚本：`experiments/aco_source_safe_1to10/run_expanded_prior_component_ablation.py`
- LDA-only formal audit：`experiments/aco_source_safe_1to10/run_expanded_lda_only_formal_audit.py`
- ACO 候选先验：`experiments/aco_source_safe_1to10/run_expanded_aco_ml_prior.py`
- Score4 posterior：`experiments/aco_source_safe_1to10/finalize_expanded_aco_ml_score4.py`
- 最终方法延时基准：`experiments/aco_source_safe_1to10/benchmark_expanded_aco_ml_prior_latency.py`
- 结果目录：`results/expanded_source_safe_1to10/aco_ml_candidate_prior/`
- 最终报告：`final_aco_score4_ml_report.json`、`final_aco_score4_ml_report.md`
- 逐包结果：`final_test_predictions.csv`
- 权重表：`validation_weight_selection.csv`、`score4_ml_beta_selection.csv`
- 延时结果：`latency/expanded_aco_ml_prior_latency.json`、`latency/expanded_aco_ml_prior_latency.md`
- 优化延时结果：`latency/expanded_aco_ml_prior_latency_optimized.json`、`latency/expanded_aco_ml_prior_latency_optimized.md`
- 纯 LDA+RF 延时：`latency/expanded_lda_rf_only_latency.json`、`latency/expanded_lda_rf_only_latency.md`
- LDA/RF 组件消融：`prior_component_ablation_fixed_alpha_beta/component_ablation_report.json`
- LDA-only formal audit：`prior_component_ablation_fixed_alpha_beta/lda_only_formal_audit/lda_only_formal_audit.json`

## 报告限制

该 Expanded test 已在旧 ACO、source-level ACO、候选排序等实验中查看，当前结果不能再
称为一次未触碰的 confirmatory test。投稿前应固定 `alpha=0.3`、`beta=0.5` 和全部代码，
再用新的独立采集、预注册重复划分或外部环境进行最终确认。
