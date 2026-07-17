# INFOCOM 2027 实验交接

更新日期：2026-07-17

## 当前主线（唯一默认口径）

当前 Expanded-649 主线是 `expanded_LDA_ACO_no_alpha`：

- 数据：`ExpandedReal-649-v1`，32 个位置、649 个 QC 后真实 source packet。
- 固定 source split：train/validation/test = `393/128/128`，seed `20260626`，source overlap 为 0。
- 仅 train 做 1:10 增强：validation 阶段为 `3930/128/128` rows；正式 refit 阶段为 `5210/128` rows。
- LDA：标准化 LDA(svd)，输入为 6 个 RSSI+ 特征和 21 个 S17/raw 特征，共 27 维。
- 候选：直接取 LDA Top-5；已经删除 RSSI/LDA 的 alpha 候选融合。
- ACO：4 segments、16 ants、12 iterations、Score4；RSSI 只保留为 ACO 观测代价和弱先验。
- 最终融合：validation 在 `beta=0.0..1.0` 上重新选择并冻结 `beta=0.6`，之后才运行 formal test。
- Validation：LDA `117/128`，完整 ACO 最终 `119/128 = 92.97%`。
- Formal：LDA `120/128`，完整 ACO 最终 `120/128 = 93.75%`；LDA Top-5 recall `128/128`。
- Formal test 在历史实验中已经被查看，因此当前 formal 数值必须标记为 exploratory。

权威入口：

- 实验说明：`docs/mainline_202607/EXPANDED_LDA_ACO_NO_ALPHA_REFREEZE_20260717.md`
- 冻结配置：`results/expanded_source_safe_1to10/aco_lda_only_no_alpha_refrozen_20260717/FROZEN_CONFIG.json`
- 入口脚本：`experiments/aco_source_safe_1to10/run_no_alpha_validation_refreeze.py`
- 结果目录：`results/expanded_source_safe_1to10/aco_lda_only_no_alpha_refrozen_20260717/`
- 完整交接步骤：`docs/mainline_202607/EXPANDED_649_PARTNER_HANDOFF_20260717.md`
- 与旧 370 主线对比：`docs/mainline_202607/EXPANDED649_VS_OLD370_MAINLINE_20260717.md`

## 数据获取

Expanded 数据以两个普通 Git 压缩包交付，不依赖旧的 `mainline_202607` LFS 包：

1. `data/expanded_real_32points_20260716/algorithm_ready/LoRaMorph_ExpandedReal649_v1_20260716.tar.gz`
   - 19 MiB；SHA-256：`21dac9b8ad448211de8faef5a3748cfc3896a4f50b025f2e9a0c00642e860210`
   - 包含未划分的 649 个真实包、RSSI+/S17/q1-q4、ML 特征、NPZ、QC 和数据字典。
2. `data/expanded_real_32points_20260716/source_safe_1to10/deliverables/ExpandedReal649_source_safe_1to10_seed20260626_partner_20260716.tar.gz`
   - 77 MiB；SHA-256：`625723f2f0a1769554c25bb9187202c41e53d8acba147f6c8b46d2c72fffa81a`
   - 包含固定 393/128/128 划分、train-only 1:10 增强、全部算法输入和验证报告。

正式 5210/128 refit 数据不重复上传；它由上述两个包使用
`experiments/aco_source_safe_1to10/build_expanded_trainval_refit.py` 确定性生成。

克隆后可在仓库根目录运行 `python3 fingerprint_localization/scripts/verify_expanded_handoff.py`，
同时检查两个压缩包的大小/SHA-256、核心文件和冻结配置。

## 已完成但不能误读的比较实验

- 候选召回/受控弱包：结果位于 `candidate_recall_and_controlled_weakness_20260716/`。它使用删除 alpha 之前的配置，适合作为诊断基线，不能直接冒充当前 no-alpha 主线结果。
- 信息素/贪心/平均代价消融：结果位于 `search_mechanism_ablation_20260717/`，同样基于旧 alpha 配置。当前结果不能证明信息素机制不可替代。
- 当前 no-alpha 的 clean 消融已经随主线一起输出：formal 的无信息素最终为 `121/128`，完整 ACO 为 `120/128`，差异不显著。
- RSSI-only LDA：formal LDA/最终 ACO 为 `112/115`，低于 RSSI+S17 的 `120/120`。该结果说明 S17 主要改善 LDA Top-1 和 posterior 质量。

## 搭档继续实验的优先级

所有新比较必须先在 validation 冻结配置，formal 只能使用冻结配置一次性评估；不要用 formal 选择阈值、seed 或规则。

1. 外部基线：KNN/probabilistic fingerprint、RF/SVM/MLP、D-Trace RSSI+、OrchLoc、MC-LoRa。
2. Solver：small-scale exhaustive search、greedy segment selection、weighted voting、random search、无信息素、完整 ACO。
3. 可扩展性：segment 数 4/8/16、Top-k、阈值、ants/iterations 的 accuracy-latency 曲线。
4. 当前 no-alpha 协议下重跑候选召回、弱包扰动和完整机制消融；旧 alpha 结果仅作对照。
5. 原始 IQ 层面的前导码缺失、幅度噪声、CFO/频移和单 segment 异常；现有 feature-space 扰动不能替代该实验。
6. 多 seed/多 split 或新增盲测包，报告 paired bootstrap/置信区间；否则不能强化“ACO 不可替代”结论。
7. 端到端时延、峰值内存和计算复杂度；统一注明是否包含特征提取、模型加载和模板构建。

## 禁止事项

- 不得重新随机划分 649 个 source packet。
- 不得在 validation/test 上拟合 scaler、增强统计、模板、阈值或候选规则。
- 不得把 3930/5210 个增强 rows 当作独立真实包样本量。
- 不得把旧 370 的 `67/74` 与 Expanded 的 `120/128` 当成同 test set 的直接提升。
- 不得再使用 `_fail` chirp 文件、`v2_output_wrong` 或增强后才切分的旧结果。
