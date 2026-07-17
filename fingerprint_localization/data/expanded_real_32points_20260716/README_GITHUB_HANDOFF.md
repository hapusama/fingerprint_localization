# ExpandedReal-649 GitHub data handoff

GitHub 只提交两个已校验的压缩包和少量可直接浏览的 metadata；没有重复提交 1.2 GiB 的解压后 source-safe/refit CSV。

| Package | Size | Purpose | SHA-256 |
| --- | ---: | --- | --- |
| `algorithm_ready/LoRaMorph_ExpandedReal649_v1_20260716.tar.gz` | 19 MiB | 未划分父数据、算法输入、特征、NPZ、QC/恢复记录 | `21dac9b8ad448211de8faef5a3748cfc3896a4f50b025f2e9a0c00642e860210` |
| `source_safe_1to10/deliverables/ExpandedReal649_source_safe_1to10_seed20260626_partner_20260716.tar.gz` | 77 MiB | 固定 393/128/128 split、train-only 1:10 增强和验证报告 | `625723f2f0a1769554c25bb9187202c41e53d8acba147f6c8b46d2c72fffa81a` |

正式 5210/128 refit 输入由 `experiments/aco_source_safe_1to10/build_expanded_trainval_refit.py` 从这两个包确定性生成。完整命令见 `docs/mainline_202607/EXPANDED_649_PARTNER_HANDOFF_20260717.md`。
