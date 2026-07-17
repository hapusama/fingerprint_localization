# Expanded Real 32-Position Dataset

- Protocol: source-packet-safe physical 5-second slots with per-file robust period fit
- Expected RSSI-linked physical slots: 661
- Final accepted real packets: 649
- Baseline packets: 370
- Retained source-safe baseline starts: 364
- Legacy unpaired starts (kept separately): 6
- Net growth: +279 (75.41%)
- Positions: 32
- Rejected or unavailable: 12

## Consistency checks

- accepted_source_keys_unique: True
- rssi_key_set_matches: True
- packet_q_key_set_matches: True
- packet_q_has_q1_q4_for_every_key: True
- spectrum_key_set_matches: True
- spectrum_rows_per_key_are_352: True
- position_count_is_32: True

## Rejection reasons

- iq_window_out_of_range: 1
- low_nonzero_fraction: 1
- low_packet_score: 1
- low_raw_power: 1
- missing_detection: 10
