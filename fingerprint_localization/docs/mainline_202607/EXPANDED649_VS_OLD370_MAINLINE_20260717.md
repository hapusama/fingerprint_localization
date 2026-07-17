# Expanded-649 与旧 370 主线差异

更新日期：2026-07-17

这两条主线的数据、位置集合、split、候选器和统计单位都不同。`67/74` 与 `120/128` 不能解释为同一测试集上的直接提升。

| 维度 | 旧 370 主线 | 当前 Expanded-649 no-alpha 主线 |
| --- | --- | --- |
| 真实 source packet | 370 | 649 |
| 位置数 | 54 | 32 |
| 每点包数 | 不均衡、整体较稀疏 | 19--21 |
| Source split | 223/73/74 | 393/128/128 |
| Seed | 20260626 | 20260626 |
| Validation/test 增强 | 主线不增强；另有探索性 TTA | 不增强 |
| Validation 训练 rows | 2230/73/74 | 3930/128/128 |
| Formal refit | 296 sources、2960 rows，test 74 | 521 sources、5210 rows，test 128 |
| 候选生成 | RSSI source-level Top-5，class k=3 | 标准化 LDA(RSSI+S17) posterior Top-5 |
| Alpha 候选融合 | 无 | 已删除；直接 LDA Top-5 |
| ACO | 4 segments、16 ants、12 iterations、Score4 | 同一 ACO 骨架和规模 |
| 最终监督融合 | 无 LDA beta 融合 | `beta=0.6` LDA posterior 融合 |
| 冻结 formal | 67/74 = 90.54% | 120/128 = 93.75% |
| 与简单候选器关系 | 相对 RSSI 候选器有净纠错 | 最终与纯 LDA 同为 120/128 |
| Packet-to-location | median 16.908 ms，P95 17.345 ms | no-alpha median 17.156 ms，P95 20.489 ms（不同运行时/批次） |
| 结果状态 | 预先存在的 74 包冻结结果可作为历史确认性结果 | formal 已被多次查看，只能标为 exploratory |

## 数据构建差异

旧 370 数据来自早期严格对齐包集合。其 54-point 输入文件与当前 Expanded 的 32-position 数据不是简单的子集/超集关系。Expanded 重新执行 packet-slot 恢复、RSSI/IQ 配对和 QC；其中 6 个旧 dense starts 超出 RSSI 可配对物理槽范围，没有进入新的严格配对集合。

Expanded 的 649 个包先形成未划分父数据，再按 source packet 划分；增强统计、scaler、模板和阈值都必须从 train source 拟合。旧 370 的早期 Gaussian pool 曾在 split 前用全数据列标准差估计噪声尺度，因此旧 TTA 结果带有轻度无标签 transductive preprocessing，只能作探索性补充。

## 算法意义差异

旧主线主要回答：在 RSSI Top-5 候选较弱时，segment ACO 是否能相对 RSSI baseline 纠错。Expanded 主线加入判别力很强的 LDA 候选器后，主要瓶颈发生变化：

- LDA formal Top-1 已是 120/128，Top-5 recall 为 128/128；
- 完整 ACO search-only 只有 105/128，最终依赖 `beta=0.6` 回到 120/128；
- 因而 Expanded 主线目前不能声称 ACO 在 accuracy 上超过 LDA；
- ACO 的可辩护价值是候选约束、segment 证据、异常诊断和可解释路径，是否不可替代仍需新实验。

## 公平对比要求

若要比较算法而不是比较数据集，必须在同一个 Expanded split 上重跑旧方法，并保持：

1. 相同 393/128/128 source；
2. 相同 train-only 1:10 增强和 5210/128 refit；
3. 相同候选 Top-k 或同时报告候选 recall；
4. 相同特征可用性和时延边界；
5. 相同逐包误差、W2R/R2W 和 paired significance。

论文中可以并列报告两条历史成绩，但必须注明 test set 不同，不应计算 `93.75%-90.54%` 作为算法增益。
