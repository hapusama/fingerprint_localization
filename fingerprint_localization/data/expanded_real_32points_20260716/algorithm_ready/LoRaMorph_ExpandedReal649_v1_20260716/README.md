# LoRaMorph ExpandedReal-649-v1 Algorithm-Ready Dataset

This package contains the pooled, unsplit 32-position real-packet dataset used after 2026-07-16.

## Scope

- Real source packets: 649
- Positions: 32
- Packets per position: 19--21
- Symbols per packet: 16
- q1 raw offsets: -2, -1, 0, +1, +2
- q4 offsets: -2.00 to +2.00 in 0.25-bin steps
- S17: 17 peak-aligned bins, 8 preamble symbols after skipping symbol 0
- Split status: none; all packets are pooled

## Recommended files

- `inputs/rssi_plus_packet_level_32points_649.csv`: canonical packet-level RSSI input.
- `inputs/lora_frequency_s17_32points_649.csv`: S17 input for RF/MLP and PGAR compatibility.
- `inputs/subbin_spectrum_long_q1q4_32points_649.csv`: canonical ACO v2/v4 long spectrum input.
- `features/ml_feature_matrix_649.csv`: unscaled 27-feature matrix for 1-NN/KNN/RF/XGBoost/MLP.
- `features/pgar_packet_features_649.csv`: packet-level PGAR features.
- `features/symbol_features_10384.csv`: 649 x 16 symbol-level observations.
- `arrays/loramorph_expanded649_unscaled.npz`: aligned matrices and ACO tensors.
- `metadata/label_mapping.csv`: stable label IDs 0--31.
- `DATA_DICTIONARY.xlsx`: human-readable delivery summary and data dictionary.

## Leakage rule

All numeric matrices are intentionally unscaled. After creating a source-packet-safe split, fit scalers,
imputers, feature selectors, augmentation statistics, and prototypes on the training sources only. Do not
fit preprocessing on this pooled full dataset before held-out evaluation.

## Mainline compatibility

ACO v2/v4 can consume the RSSI CSV and q1/q4 spectrum long CSV directly. PGAR can consume the RSSI,
S17, and q1/q4 files. Conventional baselines can use `ml_feature_matrix_649.csv` or the corresponding
arrays in the NPZ file.

## Provenance

The package does not contain the 22.61 GiB raw IQ files. Accepted starts, source manifests, QC exclusions,
recovery reports, extraction parameters, and SHA-256 checksums are included for auditability.
