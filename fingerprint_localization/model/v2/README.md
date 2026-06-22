# V2 physics-guided fingerprint pipeline

This folder implements the first runnable version of `docs/理论值生成流程.md`.

Implemented pieces:

- Uses `model/v1/output/location_distance_54points.csv` as the 54-point distance/state table.
- Reads wideband chirp captures from `../dong/data_analysis` and estimates point-level `rho_chirp` and `tau_rms_chirp_us`.
- Reads the current best USRP fingerprint CSV and derives the S17-style summary features `C_S` and `J_S`.
  When the extractor CSV already contains `s17_c_s` and `s17_j_s`, those packet-internal values are used directly.
- Builds a point-level physics table with `rho_final`, `tau_d_us`, and coarse RSSI+ physics vector `x_phy`.
- Builds residual training data: `residual = x_real - x_phy`.
- Runs a direct fingerprint matching baseline and a physics-aware matching baseline.
- Provides a conditional residual GAN in `residual_gan.py`. No DDPM is used in v2.

Notes:

- The preferred S17 extractor command above uses 17 local bins (`-8..+8`) and records packet-internal `s17_c_s` / `s17_j_s`.
- If an older CSV without `s17_c_s` / `s17_j_s` is used, the pipeline falls back to deriving `C_S` from available bins and approximating `J_S` from per-location packet-level bin0 energy variation.

Typical commands from the project root:

```powershell
conda run -n MAML python -B data\utils\usrp_lora_preamble_features.py `
  --input data\raw\usrp `
  --glob *.bin `
  --output-csv data\processedData\usrp_preamble_fft_s17_54loc_20pkt_nonorm_relative_8sym.csv `
  --bin-count 17 `
  --max-packets-per-file 20 `
  --skip-preamble-symbols 1 `
  --feature-symbols 8 `
  --phase-mode relative `
  --normalize none `
  --periodic-extract `
  --packet-period-samples 2500000 `
  --period-search-samples 6000 `
  --verify-peak-std-max 2 `
  --verify-score-db 0 `
  --detect-symbols 4 `
  --detect-threshold-db 2 `
  --peak-std-max 1.5 `
  --scan-step-symbols 0.125 `
  --refine-step-samples 32 `
  --seed-max-scan-symbols 600 `
  --plot-packets-per-file 0

conda run -n MAML python -B model\v2\v2_pipeline.py `
  --usrp-csv data\processedData\usrp_preamble_fft_s17_54loc_20pkt_nonorm_relative_8sym.csv

conda run -n MAML python -B model\v2\residual_gan.py --epochs 100

conda run -n MAML python -B model\v2\gan_augmented_locator.py `
  --gan-epochs 50 `
  --locator-epochs 120 `
  --augment-per-sample 1

conda run -n MAML python -B model\v2\summarize_v2_results.py
```

Smoke run on the current local data:

```text
conda run -n MAML python -B model\v2\v2_pipeline.py `
  --usrp-csv data\processedData\usrp_preamble_fft_s17_54loc_20pkt_nonorm_relative_8sym.csv `
  --reuse-wideband-csv model\v2\output\v2_wideband_chirp_features.csv

fingerprint_only accuracy:      0.1758
fingerprint_only distance err:  10.98 m
physics_match accuracy:         0.2088
physics_match distance err:     9.59 m
```

The first physics-aware matcher is intentionally simple: it keeps the learned
USRP fingerprint distance as the main score and adds a small penalty for
inconsistent `rho`. A short weight scan on the current data picked
`rho_match_weight=0.75` as the default because it gave the lowest mean distance
error among `[0, 0.1, 0.25, 0.5, 0.75, 1.0]`.

Step 8 smoke test:

```text
conda run -n MAML python -B model\v2\gan_augmented_locator.py `
  --gan-epochs 3 `
  --locator-epochs 5 `
  --batch-size 128 `
  --max-samples-per-class 30 `
  --augment-per-sample 1

real_only accuracy:    0.1321
real_plus_gan accuracy: 0.4151
```

This smoke test only verifies the augmentation loop is wired end to end. It is
not a final localization result because both GAN and locator were intentionally
under-trained.
