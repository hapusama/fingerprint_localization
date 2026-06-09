# Agent Notes for loRa_loc

## Project Overview

This workspace contains LoRa/USRP localization and channel-sounding experiments.
The code is split into two main areas:

- `dong/`: GNU Radio Companion flowgraphs, generated Python flowgraphs, and
  hand-written helper scripts for USRP B210/B200 receive probes, chirp TX/RX,
  spectrogram rendering, matched-filter analysis, and synthetic-bandwidth
  multipath sounding.
- `fingerprint_localization/`: curated packet-feature CSV datasets for LoRa
  fingerprint-localization analysis. The current repo contains data and
  documentation, not the upstream feature-extraction script.

There is currently no `.git` directory at the workspace root. Do not assume Git
metadata, branches, or history are available unless one is added later.

## Environment

The radio workflows expect a GNU Radio/UHD Python environment. Existing docs use
Radioconda on Windows:

```powershell
& 'D:\mysoft2\radioconda\python.exe' <script.py>
gnuradio-companion <flowgraph.grc>
```

Typical Python dependencies are `numpy`, `Pillow`, GNU Radio modules
(`gnuradio`, `uhd`, `blocks`, `gr`), and GUI/runtime packages such as `PyQt5`,
`sip`, and `pmt` for generated flowgraphs.

Default inputs and outputs are kept inside `dong/`:

- `dong/inputs/`: generated or supplied local input waveforms.
- `dong/outputs/captures/`: raw IQ captures.
- `dong/outputs/analysis/`: Chime-style PNG/CSV/JSON analysis outputs.
- `dong/outputs/multipath/`: stepped-frequency sweep files and analysis runs.

Avoid treating large capture artifacts (`*.bin`, `*.npz`, generated `*.png`,
run output folders) as source files.

Open GRC files and run README commands from the workspace root when using the
checked-in `.grc` relative paths, so they resolve into `dong/inputs` and
`dong/outputs`.

## Important Hardware Defaults

- Center frequency used by most current LoRa/chirp scripts: `487.7 MHz`.
- Chime-style low-rate TX default: USRP `serial=2512552`, `2 Msps`, `2 MHz` RF
  bandwidth, `10 dB` TX gain, waveform
  `dong/inputs/chime_test_tx_period_fc32.bin`.
- Chime-style low-rate RX default: USRP `serial=2603160`, `2 Msps`, `2 MHz` RF
  bandwidth, `20 dB` RX gain, output
  `dong/outputs/captures/chime_test_rx_fc32.bin`.
- `dong/rx_power_probe.py` defaults to `serial=30AA048`, `487.7 MHz`,
  `500 kS/s`, `200 kHz` bandwidth, normalized gain `0.6`.
- `dong/multipath/capture_sweep.py` leaves `--device-args` empty by default so
  UHD auto-selects the only attached USRP. It defaults to `sc16` wire format to
  avoid weak all-zero IQ captures.

Before any over-the-air test, verify local lab/regulatory permissions, frequency
plan, gain, antennas, attenuation, and the correct USRP serials. For first
debugging, use cabled loopback plus adequate attenuation.

## Main Workflows

Generate the repeated Chime-style TX chirp waveform:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\TX\chime_wideband_make_tx_waveform.py'
```

Run low-rate Chime-style TX/RX:

```powershell
gnuradio-companion .\dong\TX\chime_wideband_tx.grc
gnuradio-companion .\dong\RX\chime_wideband_rx.grc
```

Start RX before TX. The default RX capture is 10 seconds at `2 Msps` `fc32`
and is roughly 160 MB.

Render and analyze a received Chime-style capture:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\RX\chime_wideband_spectrogram.py'
& 'D:\mysoft2\radioconda\python.exe' '.\dong\RX\chime_wideband_analyze.py'
```

Synthetic-bandwidth multipath sweep and analysis:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\multipath\capture_sweep.py' --output '.\dong\outputs\multipath\cal_loopback.npz' --tx-enable --center 487.7e6 --span 2e6 --step 125e3 --samp-rate 500e3 --rf-bandwidth 500e3 --duration 0.08 --rx-gain 0.5 --tx-gain 0.15 --tx-amplitude 0.05
& 'D:\mysoft2\radioconda\python.exe' '.\dong\multipath\analyze_sweep.py' --input '.\dong\outputs\multipath\cal_loopback.npz' --output-dir '.\dong\outputs\multipath\cal_loopback_analysis'
```

For OTA measurements, keep the calibration and measurement sweep grids exactly
the same, then pass `--calibration` to `analyze_sweep.py`.

## Code Organization Notes

- `*.grc` files are the source of truth for GNU Radio flowgraphs. Matching
  generated `*.py` files are checked in for convenience. When changing a
  flowgraph, prefer editing the `.grc` in GNU Radio Companion and regenerating
  the `.py`; keep both in sync.
- Hand-written scripts include:
  - `dong/rx_power_probe.py`: quick receive-power/FFT probe.
  - `dong/TX/chime_wideband_make_tx_waveform.py`: repeated fc32 LFM upchirp
    period generator.
  - `dong/RX/chime_wideband_spectrogram.py`: PNG spectrogram renderer with an
    expected upchirp overlay.
  - `dong/RX/chime_wideband_analyze.py`: matched-filter peak/tap analysis for
    repeated LFM chirp captures.
  - `dong/multipath/waveforms.py`: chirp-probe creation and correlation channel
    estimate helpers.
  - `dong/multipath/capture_sweep.py`: stepped-frequency USRP capture.
  - `dong/multipath/analyze_sweep.py`: frequency response, CIR/PDP CSV/PNG, and
    summary output.
- Embedded Python block files (`dong/dong_epy_block_0.py` and
  `dong/multipath/multipath_chirp_tx_epy_chirp_source.py`) are referenced by
  GRC-generated flowgraphs; preserve their class names and constructor defaults
  unless the corresponding `.grc` is updated too.

## Fingerprint Data Notes

`fingerprint_localization/data/packet_features_analysis/*.csv` contains
packet-level LoRa features. The header includes metadata fields such as
`file_name`, `lab_name`, `experiment_id`, `corridor_id`, `position_id`,
`tx_power_dbm`, SF/preamble metadata, optional packet counter, and features
such as:

- `packet_avg_power_db`: USRP IQ average packet power/strength feature, not
  SX1276 RSSI and not total packet energy.
- `preamble_peak_to_residual_db`: dechirped preamble FFT main-bin energy versus
  residual-bin energy, averaged over preamble symbols.
- `preamble_peak_width_3db_bins_avg`: average 3 dB main-lobe width in FFT bins.
- `preamble_peak_mag_bin_-8` through `preamble_peak_mag_bin_+8`: local
  main-peak-aligned FFT magnitudes, typically normalized around the main peak.

The README references an upstream extractor at
`../gr-lora_sdr/examples/lora_file_preamble_fft.py`, which is not present in
this workspace.

## Verification

There is no formal test suite in this workspace. For code-only edits, at least
run syntax compilation for changed Python files:

```powershell
$files = rg --files -g '*.py'
python -m py_compile $files
```

For scripts that use GNU Radio/UHD hardware, avoid running capture/transmit
commands as automated tests unless the user explicitly asks and the hardware
setup is confirmed. For analysis-only scripts, prefer testing with small local
sample inputs or already captured files and writing outputs under
`dong/outputs`.

## Maintenance Guidance

- Keep Windows path handling explicit. Use `pathlib.Path` in hand-written Python
  and quote PowerShell paths with spaces.
- Keep generated capture files out of source-oriented edits. If a workflow
  produces new data, document where it lives and whether it is calibration,
  loopback, or OTA measurement data.
- Maintain the distinction between normalized USRP gain and dB gain. Existing
  scripts use both conventions depending on the UHD block.
- Be cautious when changing sample rate, RF bandwidth, chirp bandwidth, and
  analyzer parameters: these values must move together for TX, RX, waveform
  generation, spectrogram rendering, and matched-filter analysis.
