# 2026-07-10 Mainline Lock

This file marks the experiment branch to use for paper writing.

## Mainline Statement

The current paper mainline is:

1. ACO v2.0 baseline.
2. ACO v4.0 main method.
3. ACO v4.7 two-stage rule branch as a comparison/validation branch.
4. Final reported protocol uses source-safe validation selection followed by train+val final refit.

## Source-Safe Data Protocol

Use the source-safe data processing protocol:

- Split original source packets first.
- Training uses only augmented copies from training source packets.
- Validation and test remain original, unaugmented packets during configuration selection.
- Configuration is selected only on the old independent validation split.
- Final refit merges original train+val source packets as the final training source and applies 1:10 augmentation.
- Test remains the original 74 source packets, is not augmented, is not used for tuning, and has zero source overlap with final training.

Evidence:

- `build_group_safe_1to10_data.py`: documents and implements source-packet-first splitting.
- `group_safe_1to10/data/group_safe_metadata.json`: old split protocol, source counts train=223, val=73, test=74, source overlap train_val=0, train_test=0, val_test=0.
- `group_safe_trainval_refit/data/refit_metadata.json`: final refit protocol, train_refit=296 source packets, test=74 source packets, train rows=2960, test rows=74, test_source_overlap_with_train=0.

## Validation-Based Configuration Selection

The old independent validation split selected:

- Method: ACO v4
- `top_k=5`
- `rssi_class_k=3`

Evidence:

- `group_safe_1to10/results_tuning/aco_v4_top5_k3/aco_v4_summary.csv`

Validation result:

- val packets: 73
- val correct: 60
- val accuracy: 0.8219178082
- val RSSI top-k recall: 0.9589041096

Compared tuning candidates:

- `aco_v4_top3_k1`: val correct 57/73, accuracy 0.7808219178
- `aco_v4_top5_k1`: val correct 57/73, accuracy 0.7808219178
- `aco_v4_top5_k3`: val correct 60/73, accuracy 0.8219178082

## Final Train+Val Refit Result

Final model uses the selected ACO v4 configuration and refits with train+val source packets:

- Method: ACO v4 top5 k3 + trainval refit
- Test packets: 74 original packets
- Test correct: 67
- Test accuracy: 0.9054054054
- Test source overlap with train: 0

Evidence:

- `group_safe_trainval_refit/results/method_summary_with_aco_v4.csv`
- `group_safe_trainval_refit/results/aco_v4_top5_k3_refit/aco_v4_summary.csv`
- `group_safe_trainval_refit/results/refit_comparison_summary.json`

## Main Comparison Numbers

Use these numbers when writing the method/result story:

- RSSI+ KNN source-safe: 50/74, accuracy 0.6756756757
- ACO v4 top3 k3 source-safe: 63/74, accuracy 0.8513513514
- ACO v4.7 two-stage source-safe: 62/74, accuracy 0.8378378378
- ACO v4 top5 k3 + trainval refit: 67/74, accuracy 0.9054054054

Evidence:

- `group_safe_trainval_refit/results/refit_comparison_summary.json`

The final refit fixes 4 samples and breaks 0 relative to the source-safe ACO v4 top3 k3 reference in this comparison summary.

## ACO v2.0 Baseline

ACO v2.0 is the baseline ACO branch.

Evidence:

- `group_safe_1to10/results/aco_v2/aco_v2_summary.csv`

Old source-safe split result:

- test packets: 74
- test correct: 63
- test accuracy: 0.8513513514

## ACO v4.0 Main Method

ACO v4.0 is the main method family. The paper final reported version is the validation-selected `top_k=5, rssi_class_k=3` configuration with train+val final refit.

Evidence:

- Configuration selection: `group_safe_1to10/results_tuning/aco_v4_top5_k3/aco_v4_summary.csv`
- Final refit: `group_safe_trainval_refit/results/aco_v4_top5_k3_refit/aco_v4_summary.csv`

## ACO v4.7 Branch Boundary

ACO v4.7 is a two-stage rescue/rule branch, not the final refit result.

Evidence:

- `group_safe_1to10/results/aco_v47_two_stage_rules/aco_v47_final_summary.csv`
- `group_safe_1to10/results/aco_v47_two_stage_rules/aco_v47_metrics.json`

Old source-safe split result:

- val correct: 60/73, accuracy 0.8219178082
- test correct: 62/74, accuracy 0.8378378378
- selected rule order: `F_knnmfr_trans05;B_v43raw_score0`

Do not describe v4.7 as participating in the final train+val refit unless a new refit run is created. There is no v4.7 output under `group_safe_trainval_refit`.

## Paper-Writing Boundary

For the paper mainline:

- Use old independent val only for configuration selection.
- Use final train+val refit only after selection is frozen.
- Report final test result from `group_safe_trainval_refit`.
- Do not tune on test.
- Do not augment test.
- Do not mix old source-safe split test numbers with final refit test numbers without clearly labeling them.
- Treat ACO v4.7 as a comparison/validation branch unless rerun under the final refit protocol.
