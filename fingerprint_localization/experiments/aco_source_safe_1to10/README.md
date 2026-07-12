# Source-safe ACO mainline

主线入口：

- `run_experiment.py`：生成 1:10 noisy pool；其旧 split 结果不用于论文。
- `build_group_safe_1to10_data.py`：source-first train/validation/test。
- `build_trainval_refit_data.py`：生成最终 train+validation refit 数据。
- `run_aco_v2_on_split.py`：ACO v2 基线。
- `run_aco_v4_on_split.py`：ACO v4 调参与最终测试。
- `run_aco_v4_lora_only_ablation.py`：no-chirp 消融。

搭档交接说明见 `../../HANDOFF.md`。完整参数和命令见
`../../docs/mainline_202607/EXPERIMENT_MAINLINE_SINCE_20260623.md`。
