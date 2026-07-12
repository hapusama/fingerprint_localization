# Anchor-ACO 3.0

This folder contains the Anchor-ACO trial from
`external_design_notes/蚁群算法3.0.md`.

## Data Policy

- Default inputs are the fixed `fingerprint_localization/experiments/aco_source_safe_1to10` augmented data.
- The script consumes existing noisy CSVs and `split_assignments.csv`.
- It does not regenerate Gaussian noise and does not create a new split.

## Model

Anchor-ACO changes the packet path state from segment-wise candidate switching
to a packet-level anchor plus segment reliability flags:

- `L`: one RSSI+ Top-K candidate treated as the whole-packet anchor.
- `g_s = 0`: segment `s` supports `L`.
- `g_s = 1`: segment `s` is abnormal and is excluded from anchor evidence.

The default run uses RSSI+ Top-3, 4 preamble segments, and at most 1 abnormal
segment per packet.

## Reproduce

Run from the repository root:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/anchor_aco/anchor_aco_v3.py
```

Default output:

- `output_gaussian_noise_1to10/anchor_aco_v3_summary.csv`
- `output_gaussian_noise_1to10/anchor_aco_v3_metrics.json`
- `output_gaussian_noise_1to10/<split>_predictions.csv`
- `output_gaussian_noise_1to10/<split>_candidate_scores.csv`
- `output_gaussian_noise_1to10/<split>_segment_costs.csv`

## Current Result

| split | RSSI+ Top3 recall | anchor cost | robust best3 | anchor pheromone | anchor vote |
| --- | ---: | ---: | ---: | ---: | ---: |
| train LOOCV | 0.9730 | 0.8820 | 0.8847 | 0.8829 | 0.8829 |
| validation | 0.9784 | 0.8851 | 0.8892 | 0.8878 | 0.8919 |
| test | 0.9716 | 0.8865 | 0.8838 | 0.8824 | 0.8811 |

The current best test output is `anchor_cost`: `656/740 = 0.8865`.
