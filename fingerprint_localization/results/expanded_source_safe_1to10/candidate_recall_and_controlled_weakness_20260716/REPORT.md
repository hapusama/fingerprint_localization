# Expanded LDA/ACO candidate recall and controlled weakness audit

Date: 2026-07-16

## Candidate recall

| Split | RSSI R@1/3/5 | LDA R@1/3/5 | fused R@5 | variable union recall | mean union length | unrecoverable truncations |
| --- | --- | --- | --- | --- | --- | --- |
| validation | 92/115/122 | 117/126/128 | 127/128 | 128/128 | 6.71 | 1 |
| formal_test | 89/118/121 | 120/128/128 | 128/128 | 128/128 | 6.73 | 0 |

The variable union is `LDA Top-5 union RSSI Top-5` (deduplicated, length 5--10).
An unrecoverable truncation is a packet whose true label is absent from the fused Top-5.

## Gated ACO calibration

The gate accepts a fixed-fusion correction over LDA only when `Q_seg >= 0.848374`. This threshold was selected only on validation: 119/128 correct, 2 accepted corrections, correction precision 100.00%.

## Strongest artificial degradation

| Condition | Method | Accuracy | topology MAE / P95 (m) | severe >10 m | correction precision |
| --- | --- | --- | --- | --- | --- |
| preamble_missing (4.0) | LDA | 76.56% | 1.02 / 5.52 | 1.56% | n/a |
| preamble_missing (4.0) | Fixed fusion | 82.81% | 0.76 / 3.39 | 1.56% | 69.23% |
| preamble_missing (4.0) | Gated ACO | 78.12% | 0.97 / 5.52 | 1.56% | 100.00% |
| amplitude_noise (1.0) | LDA | 16.41% | 10.66 / 27.08 | 52.34% | n/a |
| amplitude_noise (1.0) | Fixed fusion | 17.97% | 10.71 / 27.01 | 52.34% | 9.52% |
| amplitude_noise (1.0) | Gated ACO | 16.41% | 10.71 / 27.08 | 51.56% | 0.00% |
| cfo_shift (1.0) | LDA | 77.34% | 1.20 / 6.67 | 3.12% | n/a |
| cfo_shift (1.0) | Fixed fusion | 79.69% | 1.02 / 6.67 | 0.78% | 50.00% |
| cfo_shift (1.0) | Gated ACO | 77.34% | 1.12 / 6.67 | 2.34% | 40.00% |
| segment_anomaly (1.0) | LDA | 83.59% | 0.86 / 6.67 | 0.78% | n/a |
| segment_anomaly (1.0) | Fixed fusion | 82.03% | 0.89 / 6.67 | 1.56% | 38.46% |
| segment_anomaly (1.0) | Gated ACO | 82.03% | 0.99 / 6.67 | 1.56% | 0.00% |

## Definitions and limitations

- Topology error is `abs(distance_m(prediction) - distance_m(truth))`; it is not Euclidean error.
- Severe error is fixed at topology error >10 m (about three sampling intervals).
- Correction precision is beneficial LDA corrections divided by all predictions changed from LDA.
- Coverage-risk selects the most confident 10%--100% of packets using each method's native margin.
- Low detect score, low SNR and high segment-cost standard deviation use validation-quartile thresholds.
- Preamble loss, amplitude noise, CFO shift and a single-segment anomaly are feature-space proxies. They do not replace a raw-IQ channel/noise injection study.

Validation diagnostic cut points:

```json
{
  "detect_score_db": {
    "q25": 2.7651140689849854,
    "q50": 5.423491716384888,
    "q75": 8.254794359207153
  },
  "snr": {
    "q25": 8.0,
    "q50": 9.0,
    "q75": 9.0
  },
  "segment_cost_std": {
    "q25": 0.0006427756925840823,
    "q50": 0.0020341939894012516,
    "q75": 0.0042885497277126905
  }
}
```

## Reproduce

```bash
python fingerprint_localization/experiments/aco_source_safe_1to10/run_candidate_recall_and_controlled_weakness.py
```
