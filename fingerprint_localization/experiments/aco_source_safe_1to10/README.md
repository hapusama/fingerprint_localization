# Source-safe ACO mainline

主线入口：

- `run_experiment.py`：生成 1:10 noisy pool；其旧 split 结果不用于论文。
- `build_group_safe_1to10_data.py`：source-first train/validation/test。
- `build_trainval_refit_data.py`：生成最终 train+validation refit 数据。
- `build_expanded_source_safe_1to10.py`：从 Expanded-649 父数据生成固定 393/128/128 source-safe 数据。
- `build_expanded_trainval_refit.py`：在配置冻结后生成 Expanded 5210/128 formal refit 数据。
- `run_expanded_lda_aco_mainline.py`：训练 RSSI+S17 LDA 并生成删除 alpha 前的阶段性结果/模型。
- `run_no_alpha_validation_refreeze.py`：当前权威入口；直接 LDA Top-5，validation 重选 beta 后运行 formal。
- `run_candidate_recall_and_controlled_weakness.py`：候选召回、诊断分组、弱包扰动和 coverage-risk。
- `run_search_mechanism_ablation.py`：平均代价、贪心、无信息素与完整 ACO 消融。
- `run_rssi_only_lda_no_alpha_refreeze.py`：LDA 仅使用 6 维 RSSI+ 的受控特征消融。
- `run_aco_v2_on_split.py`：ACO v2 基线。
- `run_aco_v4_on_split.py`：ACO v4 调参与最终测试。
- `run_aco_v4_lora_only_ablation.py`：no-chirp 消融。

搭档交接说明见 `../../HANDOFF.md`。完整参数和命令见
`../../docs/mainline_202607/EXPERIMENT_MAINLINE_SINCE_20260623.md`。
