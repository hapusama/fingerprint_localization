# Expanded LDA/ACO 去除 alpha 融合后的 validation 重新冻结

## 变更

原主线的候选排序使用：

```text
C_candidate = (1-alpha) * C_RSSI + alpha * scaled(-log P_LDA)
```

本轮完全删除该 alpha 融合，不再扫描或保留 alpha 参数。新的候选策略固定为：

- 候选集：直接 LDA Top-5。
- RSSI：仅作为 ACO 内部观测代价和弱先验，不与 LDA 做候选排序融合。
- ACO 的 ants、iterations、Score4、segment 代价、`T_seg` 和 seed 全部保持 2026-07-16 主线配置。
- alpha 不参与 LDA 模型训练，因此 validation train-only LDA 和 formal train+validation refit LDA 无需重训；其模型参数本身没有过期。

## validation-only 冻结

在加载 formal context 前，仅用 validation 在 `beta=0.0,0.1,...,1.0` 上选择完整 ACO 最终融合权重。选择规则是最大化 validation accuracy，同分时选更小 beta。

| beta | validation 正确数 | 相对 LDA 改变数 | 有益/有害改正 |
| ---: | ---: | ---: | ---: |
| 0.0--0.3 | 106/128 | 21 | 4/15 |
| 0.4 | 108/128 | 19 | 4/13 |
| 0.5 | 114/128 | 10 | 3/6 |
| **0.6** | **119/128** | **2** | **2/0** |
| 0.7--0.8 | 118/128 | 1 | 1/0 |
| 0.9--1.0 | 117/128 | 0 | 0/0 |

最终冻结为 `beta=0.6`。formal 只使用这一个已冻结 beta，没有使用 formal 结果二次选择。

## 候选集结果

| Split | 原 alpha=0.3 Top-5 recall | 新 LDA Top-5 recall | 候选列表变化包数 |
| --- | ---: | ---: | ---: |
| validation | 127/128 | **128/128** | 89/128 |
| formal | 128/128 | **128/128** | 85/128 |

去除 alpha 后 validation 不再有候选截断错误。

## 冻结后结果

| Split | 方法 | search/final 正确数 | final accuracy | MAE/P95 (m) | 严重错误率 | correction precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | 平均代价 | 105/119 | 92.97% | 0.472/3.390 | 3.12% | 100% |
| validation | 贪心路径 | 105/119 | 92.97% | 0.472/3.390 | 3.12% | 100% |
| validation | 无信息素 | 105/119 | 92.97% | 0.472/3.390 | 3.12% | 100% |
| validation | 完整 ACO | **106/119** | **92.97%** | 0.472/3.390 | 3.12% | 100% |
| formal | 平均代价 | 104/120 | 93.75% | 0.315/3.390 | 0.78% | n/a |
| formal | 贪心路径 | 104/120 | 93.75% | 0.315/3.390 | 0.78% | 50% |
| formal | 无信息素 | 103/**121** | **94.53%** | **0.289/2.204** | 0.78% | 100% |
| formal | 完整 ACO | **105**/120 | 93.75% | 0.315/3.390 | 0.78% | 50% |

## 与原 alpha=0.3 主线比较

| Split | 原/新完整 ACO final | 标签变化 | W2R/R2W | McNemar p |
| --- | ---: | ---: | ---: | ---: |
| validation | 117/119 | 5 | 3/1 | 0.625 |
| formal | 120/120 | 4 | 2/2 | 1.000 |

删除 alpha 后，完整 ACO 的 formal final accuracy 与原主线完全持平；validation 多对 2 包，但差异不显著。

需要区分 search-only 与 final：新的完整 ACO search-only 是 validation `106/128`、formal `105/128`，而旧 alpha=0.3 主线分别为 `117/128`、`118/128`。这说明 alpha 曾经把 LDA 信息提前注入 ACO Score4；去除 alpha 后，最终 `120/128` 主要由冻结的 `beta=0.6` 后验融合恢复。

## 结论

- alpha 候选融合可以删除；完整 ACO formal 结果不降，仍为 `120/128 = 93.75%`。
- validation 冻结配置为“LDA Top-5 + beta=0.6 + 原 ACO 参数”。
- 完整 ACO 在纯 search 阶段仍是四路方法中最好，但最终融合后无信息素多对 1 包；该差异的配对检验 `p=1.0`，不能认为无信息素显著更优。
- 四种路径方法在冻结 beta 下的输出过于接近，信息素不可替代性仍未建立。

formal test 在早期实验中已被查看，本轮 formal 结果仍为 exploratory。

## 复现

```bash
PYTHONPATH=/tmp/fingerloc_py39_20260716:fingerprint_localization/experiments/aco_source_safe_1to10 \
  /usr/bin/python3 -B \
  fingerprint_localization/experiments/aco_source_safe_1to10/run_no_alpha_validation_refreeze.py
```

输出目录：`fingerprint_localization/results/expanded_source_safe_1to10/aco_lda_only_no_alpha_refrozen_20260717/`
