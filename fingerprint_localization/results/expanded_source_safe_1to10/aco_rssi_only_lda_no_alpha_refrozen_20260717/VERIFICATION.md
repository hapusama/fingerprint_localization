# Verification

Independent rerun directory: `/private/tmp/rssi_only_lda_no_alpha_rerun`.

- Validation beta selection, candidate recall, paired comparison, frozen configuration, and full-ACO pairwise tables were byte-identical.
- Validation and formal packet keys, candidate lists, LDA labels, search labels, final labels, and frozen beta were identical.
- Saved validation and formal model files were byte-identical across runs.
- Both saved pipelines report `n_features_in_ = 6` for the pipeline, scaler, and LDA estimator.
- The six model inputs exactly match `RSSI_COLUMNS`; all 21 `RAW_COLUMNS`, including the 17 spectral bins and S17 diagnostics, are excluded from LDA.
- ACO parameter dictionaries and validation/formal segment thresholds are exactly equal to the RSSI+S17 no-alpha reference.
- Runtime used for the official run: NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.3.2, joblib 1.3.2.
