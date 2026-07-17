# ExpandedReal-649-v1-source-safe-1to10-v1

This dataset is derived from `ExpandedReal-649-v1`.

## Split

- Seed: 20260626
- Train sources: 393
- Validation sources: 128
- Test sources: 128
- Source overlap: 0 for every split pair
- All 32 positions occur in train, validation, and test

## Augmentation

- Only training sources are augmented.
- Each train source produces 10 Gaussian-perturbed copies.
- Augmented train rows: 3930
- Validation rows: 128 untouched originals
- Test rows: 128 untouched originals
- Noise statistics are estimated from the 393 training sources only.
- Spectrum augmentation keeps q1/q4 integer bins and magnitude/dB/complex fields consistent for training copies.
- Validation/test rows are byte-for-field copies of the parent packet tables; parent q1/q4 differences are not rewritten.

## Algorithm inputs

- `data/noisy_rssi_plus_packet_level_32points_649.csv`
- `data/noisy_lora_frequency_s17_32points_649.csv`
- `data/noisy_subbin_spectrum_long_32points_649.csv`
- `data/split_assignments.csv`
- `data/source_packet_split.csv`
- `arrays/source_safe_1to10_arrays.npz`
- `features/source_safe_1to10_ml_features.csv`
- `metadata/validation_report.json`
- `reports/ExpandedReal649_SourceSafe_1to10_Report.xlsx`

The NPZ contains raw and train-standardized RSSI, RSSI+S17, and PGAR matrices, plus q1/q4 symbol tensors.
Do not use validation or test data to refit scalers, augmentation statistics, thresholds, or prototypes.

## Validation

The independent validation report has `status: PASS`. It confirms zero source overlap, exact preservation of
all 256 validation/test originals, train-only scaling, 352 spectrum rows per packet, and successful PGAR/ACO v2
loading of all 4,186 packets. Full revalidation of evaluation-field identity also requires the parent
`LoRaMorph_ExpandedReal649_v1_20260716` package.
