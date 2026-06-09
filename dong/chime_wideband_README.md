# Chime-Style Chirp Sounding

Current default profile is a low-rate bench test for validating the full
TX/RX flow and visualizing the LoRa-like upchirp ridge before increasing
bandwidth.

Current USRP assignment:

- TX: `serial=2512552`
- RX: `serial=2603160`

Swap the `tx_device_addr` / `rx_device_addr` variables in GRC if your cables
are connected the other way around.

## Low-Rate Test Parameters

TX:

- Center frequency: 487.7 MHz
- Sample rate: 2 Msps
- RF bandwidth: 2 MHz
- TX gain: 10 dB
- Waveform: 1 MHz LFM upchirp, -500 kHz to +500 kHz baseband
- Chirp duration: 1 ms
- Repeat period: 20 ms
- Waveform file: `.\dong\inputs\chime_test_tx_period_fc32.bin`

RX:

- Center frequency: 487.7 MHz
- Sample rate: 2 Msps
- RF bandwidth: 2 MHz
- RX gain: 20 dB
- Output: `.\dong\outputs\captures\chime_test_rx_fc32.bin`

## Run

Run these commands from the workspace root so the GRC relative file paths
resolve into `.\dong\inputs` and `.\dong\outputs`.

Generate the TX waveform file once:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\TX\chime_wideband_make_tx_waveform.py'
```

Open two GRC windows:

```powershell
gnuradio-companion .\dong\TX\chime_wideband_tx.grc
gnuradio-companion .\dong\RX\chime_wideband_rx.grc
```

Start RX first, then TX.

Default RX capture is 10 seconds. At 2 Msps `fc32`, this is about 160 MB.
This profile is meant to avoid overflow while checking that the transmitted
upchirp has the expected time-frequency slope.

Render a spectrogram after capture:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\RX\chime_wideband_spectrogram.py'
```

Analyze after capture:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\RX\chime_wideband_analyze.py'
```

Outputs:

- `.\dong\outputs\analysis\chime_test_spectrogram.png`
- `.\dong\outputs\analysis\chime_test_paths.csv`
- `.\dong\outputs\analysis\chime_test_summary.json`

## Scaling Back Up

After the low-rate profile is stable, increase these together:

- TX/RX `samp_rate`
- TX/RX `rf_bandwidth`
- TX waveform `--fs`
- TX waveform `--chirp-bw`
- analyzer/spectrogram `--fs`
- analyzer/spectrogram `--chirp-bw`

For the original wideband target, use 20 Msps sample rate, 20 MHz RF bandwidth,
and a 10 MHz chirp bandwidth.
