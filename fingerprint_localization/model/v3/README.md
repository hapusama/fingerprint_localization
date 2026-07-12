# V3 heuristic PGAR experiment

This folder contains the first heuristic algorithm trial based on
`external_design_notes/启发式算法流程.md`.

Data policy:

- Allowed: already computed data under `fingerprint_localization/v2_output`,
  including `v2_output/20260623_from_raw` and later analysis folders.
- Not allowed: any data under `fingerprint_localization/v2_output_wrong`.
- This v3 script does not read v2 model-result files; it uses raw-data-derived
  feature CSVs as inputs.

Implemented method:

- RSSI+ first-stage candidate generation with default `K=3`.
- Physics-guided raw-bin structure features from q=1 bin `[-2, -1, 0, +1, +2]`:
  `E0`, `C_peak`, `R_side`, `R_asym`, `S_L`, `S_R`.
- Per-location robust prototypes using median and IQR.
- Candidate-only reranking inside RSSI+ Top-K.
- Gate logic:
  - if RSSI+ Top-1 margin is high, keep RSSI+ Top-1;
  - otherwise use raw structure only when the packet peak is reliable.
  - q=4 shape only participates after raw rerank when all q4 gates pass:
    raw Top-1/Top-2 are still close, the packet q4 curve is stable, and the
    q4 prototypes of the current candidate pair are discriminable.

q=4 scoring:

- The q=4 curve is built from `mag_db_rel_peak` in
  `subbin_spectrum_long.csv`.
- For each packet, the q4 prototype curve is the median curve over preamble
  symbols.
- `q4_stability` is the median per-offset IQR over preamble symbols.
- q4 is added with a small `gamma` only when the gate opens; otherwise
  `gamma = 0` for that packet.

Additional ACO trial:

- `aco_packet_path.py` implements packet-internal ant colony optimization.
- One packet is treated as a 16-layer virtual path, one layer per preamble
  symbol.
- Each layer chooses among RSSI+ Top-K candidate locations.
- Switching candidates inside one packet is penalized, so the ant path favors
  evidence consistency while tolerating a few bad symbols.
- Segment observation cost combines RSSI+, raw peak energy, raw structure, and
  gated q=4 shape.
- The script reports three packet-level decisions: best-path mode, self
  pheromone, and elite-ant vote.

Global multipath-field ACO trial:

- `mfr_aco_global_multipath.py` implements MFR-ACO from
  `external_design_notes/全局多径蚁群.md`.
- It builds a global chirp multipath field
  `z=[log1p(K), log1p(Pdiff), log1p(tau_rms)]`, linearly fills missing chirp
  points within the same corridor and visibility state, then standardizes the
  field.
- RSSI+ still generates Top-K candidates.
- Whole-packet q=1 raw bin `[-2,+2]` is log-shape normalized and scored once per
  path mode; four preamble groups only produce raw stability `Q_W`.
- The ACO search depth is `H=4`, interpreted as candidate-consistency search
  depth, not four independent physical observations.
- The script records path-mode, self-loop pheromone, elite-vote, and physical
  posterior-score outputs.

ACO 2.0 chirp-bin trial:

- `aco_packet_path_v2.py` implements the changes from
  `external_design_notes/蚁群算法2.0.md`.
- It keeps RSSI+ Top-3 candidate generation and merges the 16 preamble symbols
  into 4 packet-internal noisy observations.
- It adds chirp-generated LoRa bin `[-2,+2]` templates, empirical/chirp
  shrinkage, variance shrinkage, Huber Gaussian bin likelihood, a garbage
  state, evidence-driven switch penalties, and reliability-weighted self-loop
  pheromone.
- Because the current chirp-bin template is not yet a stable strong
  discriminator inside Top-3, the default keeps the original energy/raw
  observation terms and uses the 2.0 bin likelihood as a weak auxiliary term.
  The best small sweep used `w_R=0.45`, `w_E=0.20`, `w_W=0.55`,
  `w_bin=0.02`, and `w_Q=0`.

Anchor-ACO 3.0 trial:

- `anchor_aco/anchor_aco_v3.py` implements the state redesign from
  `external_design_notes/蚁群算法3.0.md`.
- It changes the path state from segment-wise candidate switching to a
  packet-level anchor `L` plus per-segment normal/abnormal flags `g_s`.
- The default command consumes the fixed `fingerprint_localization/experiments/aco_source_safe_1to10` enhanced
  train/validation/test split and does not regenerate noise or split files.
- Current default test result: anchor-cost output `656/740 = 0.8865`.

ML-ACO 2.0 trial:

- `aco_v2_ml/ml_aco_v2_ranker.py` implements the Top-3 candidate reranking
  experiment from `external_design_notes/蚁群算法2.0+ML.md`.
- Because ML evaluation should not place augmented siblings in both train and
  test, this trial creates a group-safe split where all 10 augmented copies of
  one original packet stay in the same split.
- The first pass implements a pure-Python softmax logistic ranker and a
  validation-selected conservative margin rule. LightGBM/XGBoost and MLP are
  not run in the current environment because the required ML stack is not
  installed.
- Current group-safe test result: ACO 2.0 baseline `547/740 = 0.7392`;
  pairwise logistic replacement `551/740 = 0.7446`.

Typical command from the repository root:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/pgar_heuristic.py
```

Default inputs:

- `fingerprint_localization/data/mainline_202607/inputs/rssi_plus_packet_level_54points.csv`
- `fingerprint_localization/data/mainline_202607/inputs/lora_frequency_s17_54points.csv`
- `fingerprint_localization/data/mainline_202607/external/subbin_spectrum_long.csv`

Default output:

- `fingerprint_localization/fingerprint_localization/model/v3/output/pgar_metrics.json`
- `fingerprint_localization/fingerprint_localization/model/v3/output/pgar_summary.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output/pgar_predictions.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output/raw_structure_from_20260623.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output/q4_curve_from_20260624.csv`

Useful parameter scan:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/pgar_heuristic.py \
  --scan-rssi-margin 0,0.05,0.1,0.2,0.5,1.0
```

Useful q4 gate knobs:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/pgar_heuristic.py \
  --gamma 0.25 \
  --q4-raw-margin-threshold 0.2 \
  --q4-disc-threshold 0.5
```

Run the ACO trial:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/aco_packet_path.py
```

ACO output:

- `fingerprint_localization/fingerprint_localization/model/v3/output_aco/aco_metrics.json`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco/aco_summary.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco/aco_predictions.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco/aco_candidate_scores.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco/aco_symbol_features.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco/aco_symbol_candidate_costs.csv`

Run the global multipath-field ACO trial:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/mfr_aco_global_multipath.py
```

MFR-ACO output:

- `fingerprint_localization/fingerprint_localization/model/v3/output_mfr_aco/mfr_aco_metrics.json`
- `fingerprint_localization/fingerprint_localization/model/v3/output_mfr_aco/mfr_aco_summary.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_mfr_aco/mfr_aco_predictions.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_mfr_aco/mfr_aco_candidate_scores.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_mfr_aco/global_multipath_field.csv`

Run the ACO 2.0 trial:

```bash
python3 -B fingerprint_localization/fingerprint_localization/model/v3/aco_packet_path_v2.py
```

ACO 2.0 output:

- `fingerprint_localization/fingerprint_localization/model/v3/output_aco_v2/aco_v2_metrics.json`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco_v2/aco_v2_summary.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco_v2/aco_v2_predictions.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco_v2/aco_v2_candidate_scores.csv`
- `fingerprint_localization/fingerprint_localization/model/v3/output_aco_v2/aco_v2_segment_costs_and_templates.csv`
