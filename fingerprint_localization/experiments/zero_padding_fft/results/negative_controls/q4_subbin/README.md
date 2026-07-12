# Chirp-LoRa q=4 sub-bin scan

This control compares q=4 zero-padded sub-bin curves between cached chirp projection and measured LoRa.

- `full`: uses all sub-bins within `[-K,+K]`.
- `side`: removes only the center sub-bin `0.00`.
- Metrics: peak-normalized magnitude cosine and relative-to-local-peak dB Pearson.
