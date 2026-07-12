# ML-ACO 2.0

This folder contains the `蚁群算法2.0+ML.md` experiment: a lightweight
learning-to-rank module on top of ACO 2.0 evidence.

## Data Policy

- Inputs are still the 1:10 Gaussian-noise augmented data.
- For ML, this folder creates a group-safe split:
  all augmented copies of the same original packet stay in the same split.
- The existing augmented split is audited for reference, but is not used for
  ML training because augmented siblings cross train/validation/test.

## Method

The implemented first pass is a pure-Python packet-wise softmax logistic
ranker. It does not use candidate location IDs or location one-hot features.

Feature groups:

- RSSI+ rank/cost/margin.
- ACO 2.0 vote, self pheromone, path-mode flags.
- ACO 2.0 observation costs: `C_E`, `C_W`, `C_bin`, `C_Q`, `C_obs`.
- raw/chirp indicators from the ACO 2.x diagnostic feature builder.
- path/garbage summary features.

LightGBM/XGBoost and MLP are recorded as unavailable in this environment
because the current Python environment has no `numpy`, `sklearn`, `lightgbm`,
`xgboost`, or deep-learning stack installed.

## Reproduce

Run from the repository root:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/aco_v2_ml/ml_aco_v2_ranker.py
```

The first run generates ACO 2.0 features under
`output_gaussian_noise_1to10_group_safe/aco_v2_features`. Later runs reuse
those features unless `--force-aco` is passed.

Run the pairwise ACO-winner correction model:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/aco_v2_ml/pairwise_correction_model.py
```

## Outputs

- `output_gaussian_noise_1to10_group_safe/group_safe_split_assignments.csv`
- `output_gaussian_noise_1to10_group_safe/aco_v2_features/`
- `output_gaussian_noise_1to10_group_safe/ml_candidate_features.csv`
- `output_gaussian_noise_1to10_group_safe/ml_aco_v2_ranker_summary.csv`
- `output_gaussian_noise_1to10_group_safe/ml_aco_v2_ablation_sweep.csv`
- `output_gaussian_noise_1to10_group_safe/ml_aco_v2_predictions.csv`
- `output_gaussian_noise_1to10_group_safe/ml_aco_v2_feature_importance.csv`
- `output_gaussian_noise_1to10_group_safe/ml_aco_v2_ranker_metrics.json`
- `output_gaussian_noise_1to10_group_safe/pairwise_correction/pairwise_correction_dataset.csv`
- `output_gaussian_noise_1to10_group_safe/pairwise_correction/pairwise_threshold_sweep.csv`
- `output_gaussian_noise_1to10_group_safe/pairwise_correction/pairwise_summary.csv`
- `output_gaussian_noise_1to10_group_safe/pairwise_correction/pairwise_predictions.csv`
- `output_gaussian_noise_1to10_group_safe/pairwise_correction/pairwise_top_positive_features.csv`
- `output_gaussian_noise_1to10_group_safe/pairwise_correction/pairwise_top_negative_features.csv`
- `output_gaussian_noise_1to10_group_safe/pairwise_correction/pairwise_metrics.json`

## Current Result

Group-safe split audit:

- source augmented split: 366 / 370 original packets cross multiple splits.
- ML group-safe split: 0 / 370 original packets cross multiple splits.
- group-safe row counts: train 2230, validation 730, test 740.

Test results on the group-safe split:

| method | final | W2R | R2W | net |
| --- | ---: | ---: | ---: | ---: |
| E0 ACO 2.0 baseline | 547/740 = 0.7392 | 0 | 0 | 0 |
| RSSI only ranker | 479/740 = 0.6473 | 27 | 95 | -68 |
| RSSI + ACO ranker | 547/740 = 0.7392 | 0 | 0 | 0 |
| RSSI + ACO + cost ranker | 548/740 = 0.7405 | 3 | 2 | +1 |
| full logistic ranker | 544/740 = 0.7351 | 11 | 14 | -3 |
| full logistic + conservative margin | 548/740 = 0.7405 | 3 | 2 | +1 |
| pairwise logistic replacement | 551/740 = 0.7446 | 8 | 4 | +4 |

The first leak-free ML pass shows a small positive net gain only for the
cost-aware and conservative variants. The earlier fixed split remains useful
for algorithm debugging, but it is not a strict ML evaluation split because
augmented copies leak across train/validation/test.

Pairwise correction details:

- Each non-ACO-winner candidate forms one pair against the ACO vote winner.
- Pair features are `challenger - ACO winner`.
- Validation selected threshold: `0.97`.
- Test triggers: 20 replacements, with `W2R=8`, `R2W=4`, net `+4`.
