# Synthetic-Bandwidth Multipath Sounding

This folder contains a small USRP B200/B210 workflow for reproducing the
multi-path measurement idea behind the Chime paper's wideband chirp experiment.
The paper uses a wideband chirp and correlates received IQ against the known
chirp to reveal channel taps. Here we use a stepped-frequency version: capture
many narrowband frequency points, stitch the complex channel response, then
IFFT it into a synthetic-bandwidth CIR/PDP.

## What This Reproduces

- Per-frequency channel response measurement with a known chirp probe.
- Synthetic wideband response from many narrowband USRP captures.
- Power delay profile (PDP) and dominant delay taps.

This is not a drop-in clone of the paper's exact hardware setup. It is a B210
friendly reproduction path for the same measurement principle.

## Important Setup Notes

- Start with a cabled loopback plus enough attenuation before radiating.
- Leave `--device-args` empty if only one USRP is attached; this avoids serial
  mismatches like `30AA032` vs `2512552`.
- The capture script defaults to `otw_format=sc16`, not `sc8`, to avoid the
  all-zero weak-IQ problem you observed.
- Synthetic CIR phase is only meaningful after calibration. Run a loopback
  calibration sweep with the same frequency grid, then divide the measurement
  by that calibration in `analyze_sweep.py`.
- For over-the-air transmission, only use frequencies and powers allowed in
  your lab/regulatory setting.

## TX-Only GRC

For first-step transmit debugging, open this flowgraph in GNU Radio Companion:

```powershell
gnuradio-companion .\dong\multipath\multipath_chirp_tx.grc
```

It generates a periodic complex LFM chirp inside an Embedded Python source and
sends it to a B200/B210 `UHD: USRP Sink`. It stops after `tx_seconds` seconds.
Leave `device_addr` as `""` to auto-select the only attached USRP, or set it to
something like `"serial=2512552"` if several radios are attached.

Start with `tx_gain=0.10` and `amplitude=0.05`. Use a cabled loopback with
attenuation before trying antennas.
Run commands from the workspace root so relative output paths stay under
`.\dong\outputs`.

## Quick Start

From the workspace root using the `radioconda` GNU Radio environment:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\multipath\capture_sweep.py' --output '.\dong\outputs\multipath\cal_loopback.npz' --tx-enable --center 487.7e6 --span 2e6 --step 125e3 --samp-rate 500e3 --rf-bandwidth 500e3 --duration 0.08 --rx-gain 0.5 --tx-gain 0.15 --tx-amplitude 0.05
```

Then analyze it:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\multipath\analyze_sweep.py' --input '.\dong\outputs\multipath\cal_loopback.npz' --output-dir '.\dong\outputs\multipath\cal_loopback_analysis'
```

For an antenna measurement, keep the same sweep grid:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\multipath\capture_sweep.py' --output '.\dong\outputs\multipath\ota_measurement.npz' --tx-enable --center 487.7e6 --span 2e6 --step 125e3 --samp-rate 500e3 --rf-bandwidth 500e3 --duration 0.08 --rx-gain 0.5 --tx-gain 0.15 --tx-amplitude 0.05
```

Analyze with calibration:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\multipath\analyze_sweep.py' --input '.\dong\outputs\multipath\ota_measurement.npz' --calibration '.\dong\outputs\multipath\cal_loopback.npz' --output-dir '.\dong\outputs\multipath\ota_measurement_analysis'
```

Outputs:

- `frequency_response.csv`
- `frequency_response.png`
- `cir.csv`
- `pdp.png`
- `summary.json`

## Wider Synthetic Bandwidth

Delay resolution is approximately `1 / synthetic_span`.

Examples:

| Span | Nominal delay resolution | Path-length resolution |
| ---: | ---: | ---: |
| 2 MHz | 500 ns | about 150 m |
| 10 MHz | 100 ns | about 30 m |
| 20 MHz | 50 ns | about 15 m |

The step controls unambiguous delay: `1 / step`. With `125 kHz` steps, the
unambiguous delay window is about `8 us`.

To move closer to the paper's 20 MHz chirp-style experiment, try:

```powershell
& 'D:\mysoft2\radioconda\python.exe' '.\dong\multipath\capture_sweep.py' --output '.\dong\outputs\multipath\cal_20mhz.npz' --tx-enable --center 487.7e6 --span 20e6 --step 250e3 --samp-rate 1e6 --rf-bandwidth 1e6 --probe-bandwidth 400e3 --duration 0.08
```

Use the exact same grid for the measurement file and pass the calibration file
to `analyze_sweep.py`.
