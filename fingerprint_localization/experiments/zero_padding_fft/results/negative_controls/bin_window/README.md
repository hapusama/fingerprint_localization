# Chirp-LoRa bin window scan

This control scans matched-vs-mismatched similarity over LoRa bin windows.

- `full`: uses `[-K,+K]` including center bin.
- `side`: uses `[-K,-1] U [+1,+K]`, excluding center bin.
- Candidate chirp points are restricted to common chirp/LoRa locations.
- `significant_positive` requires bootstrap lower bound > 0 and sign-flip p < 0.05.
