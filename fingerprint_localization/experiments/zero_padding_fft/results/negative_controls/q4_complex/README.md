# Chirp-LoRa q=4 complex sub-bin scan

This control compares complex q=4 sub-bin curves using common-phase and linear-phase alignment.

- Complex samples are center-phase rotated and peak normalized on both sides.
- `full`: uses all sub-bins within `[-K,+K]`.
- `side`: removes only the center sub-bin `0.00`.
- Higher similarity is better; `significant_positive` tests whether matched exceeds mismatched.
