# Chime-Style Chirp Sounding

Current default profile is a 20 MS/s two-USRP bench test for validating the
full TX/RX flow and visualizing the CHIME-style wideband upchirp ridge.

Current USRP assignment:

- TX: `serial=2512552`
- RX: `serial=2603160`

For the local receiver machine, use the USRP with serial `2603160` and connect
the receive cable/antenna to its `TX/RX` input by default. Change the GRC
`rx_antenna` variable to `RX2` if the antenna is connected there. The other
radio, serial `2512552`, is the transmitter and uses its `TX/RX` port. Do not rely on "Device 0" /
"Device 1" ordering; use the serial number.

If you are not sure which physical B210 has which serial, unplug one radio and
run:

```bash
PATH=/Users/siri/radioconda/bin:$PATH uhd_find_devices
```

The remaining serial in the output identifies the connected radio. You can also
look for the serial printed on the B210 label/sticker.

Swap the `tx_device_addr` / `rx_device_addr` variables in GRC only if your
physical radios are intentionally connected the other way around.

## Wideband Test Parameters

TX:

- Center frequency: 487.7 MHz
- Sample rate: 20 Msps
- RF bandwidth: 20 MHz
- TX gain: 100 dB
- TX duration: 50 s
- Waveform: 18 MHz LFM upchirp, -9 MHz to +9 MHz baseband
- Chirp duration: 1 ms
- Repeat period: 20 ms
- Waveform file: `dong/inputs/chime_test_tx_period_fc32.bin`

RX:

- Center frequency: 487.7 MHz
- Sample rate: 20 Msps
- RF bandwidth: 20 MHz
- RX gain: 40 dB
- RX antenna/input: `TX/RX`
- Output: `/Users/siri/Desktop/data/chime_capture_<timestamp>.bin`
- Capture length: 10 s

## Run

Run these commands from the workspace root so the GRC relative file paths
resolve into `dong/inputs`.

Generate the TX waveform file once:

```bash
PATH=/Users/siri/radioconda/bin:$PATH python dong/TX/chime_wideband_make_tx_waveform.py
```

Open two GRC windows:

```bash
PATH=/Users/siri/radioconda/bin:$PATH gnuradio-companion dong/TX/chime_wideband_tx.grc
PATH=/Users/siri/radioconda/bin:$PATH gnuradio-companion dong/RX/chime_wideband_rx.grc
```

Start TX first, wait until the waterfall shows the repeated chirp, then start
RX. TX runs for 50 seconds by default so the entire 10 second RX capture falls
inside a stable transmit window.

Default RX capture is 10 seconds. At 20 Msps `fc32`, this is about 1.6 GB.

Automated no-GUI self-test:

```bash
PATH=/Users/siri/radioconda/bin:$PATH python dong/chime_wideband_tx_rx_test.py \
  --rx-seconds 10.05 \
  --max-segments 500 \
  --capture-file /Users/siri/Desktop/data/self_rx_full10_g60_r50_rx2.bin \
  --csv-out dong/outputs/analysis/self_rx_full10_g60_r50_rx2_paths.csv \
  --json-out dong/outputs/analysis/self_rx_full10_g60_r50_rx2_summary.json
```

The self-test starts TX first, waits for it to settle, captures 10.05 seconds on
RX, then runs matched-filter analysis on the first 500 chirp periods, exactly
10.00 seconds. For NLOS captures, tune TX/RX gain after first confirming a clean
LOS reference capture.

Render a spectrogram after capture:

```bash
PATH=/Users/siri/radioconda/bin:$PATH python dong/RX/chime_wideband_spectrogram.py --infile /Users/siri/Desktop/data/chime_capture_<timestamp>.bin
```

Analyze after capture:

```bash
PATH=/Users/siri/radioconda/bin:$PATH python dong/RX/chime_wideband_analyze.py --infile /Users/siri/Desktop/data/chime_capture_<timestamp>.bin --max-segments 500
```

Outputs:

- `dong/outputs/analysis/chime_test_spectrogram.png`
- `dong/outputs/analysis/chime_test_paths.csv`
- `dong/outputs/analysis/chime_test_summary.json`

For cabled tests, use suitable attenuation between the USRPs. For over-the-air
tests, use only frequencies, power levels, and shielding that are legal for the
test environment.
