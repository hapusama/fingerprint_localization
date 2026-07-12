# V4 model-driven Bayesian inversion

This folder contains a runnable first trial based on
`external_design_notes/模型驱动的贝叶斯反演算法.md`.

Data policy:

- Allowed: computed data under `fingerprint_localization/v2_output`.
- Not allowed: any data under `fingerprint_localization/v2_output_wrong`.
- This script uses raw-data-derived feature CSVs and chirp structure summaries,
  not v2 model result files.

Implemented idea:

- RSSI+ builds a Bayesian Top-K candidate set using per-location robust
  median/IQR likelihoods.
- q=4 LoRa spectra are converted into packet-level curves, then inverted into
  a low-dimensional compressed multipath state:
  `P0`, `Psec`, `tau_rms`, `K_eff`, `eta_diff`, `alpha_asym`.
- Chirp multipath summaries calibrate the relative weights of theta dimensions
  through anchor-point correlations.
- For each candidate location, the final posterior combines:
  `L_R + beta * L_E + gamma * L_Q + delta * L_theta`.
- `L_Q` is not KNN. It compares the observed q=4 curve with the candidate
  point's learned forward projection prototype while minimizing nuisance
  parameters: small sub-bin shift and affine shape scaling.
- Defaults are intentionally conservative:
  `beta = gamma = delta = 0.10`.

Typical command from the repository root:

```bash
python3 -B fingerprint_localization/experiments/baselines/bayesian_inversion/bayesian_inversion.py
```

Default inputs:

- `fingerprint_localization/data/mainline_202607/inputs/rssi_plus_packet_level_54points.csv`
- `fingerprint_localization/data/mainline_202607/inputs/lora_frequency_s17_54points.csv`
- `fingerprint_localization/data/mainline_202607/external/subbin_spectrum_long.csv`
- `fingerprint_localization/data/mainline_202607/features/chirp_point_multipath_structure_features.csv`

Default output:

- `fingerprint_localization/experiments/baselines/bayesian_inversion/output/`
