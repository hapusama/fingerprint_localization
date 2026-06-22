from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


EPS = 1e-12
FILENAME_RE = re.compile(
    r"^(?P<experiment>[^_\-]+)[_\-]"
    r"(?P<corridor>[^_\-]+)[_\-]"
    r"(?P<position>[^_\-]+)[_\-]"
    r"(?P<sf>\d+)[_\-]"
    r"(?P<tx_power>-?\d+)[_\-]"
    r"(?P<preamble>\d+)$"
)


@dataclass(frozen=True)
class FileMeta:
    file_name: str
    experiment_id: str
    corridor_id: str
    position_id: str
    sf: int
    tx_power_dbm: int
    preamble_len: int


@dataclass(frozen=True)
class LoraRuntime:
    sf: int
    sample_rate: float
    bandwidth: float
    os_factor: int
    fft_size: int
    symbol_samples: int
    downchirp: np.ndarray


@dataclass
class Candidate:
    sample_start: int
    score_db: float
    peak_bin_mean: float
    peak_bin_std: float
    peak_width_bins_avg: float
    symbol_scores_db: List[float]
    symbol_peak_bins: List[int]


class IQMemmap:
    """Read interleaved float32 I/Q samples without loading the whole file."""

    def __init__(self, path: Path):
        self.path = path
        self.raw = np.memmap(path, dtype="<f4", mode="r")
        self.num_complex = self.raw.size // 2

    def read(self, start: int, count: int) -> Optional[np.ndarray]:
        if start < 0 or count <= 0:
            return None
        stop = start + count
        if stop > self.num_complex:
            return None

        flat = self.raw[start * 2 : stop * 2]
        iq = flat.reshape(-1, 2)
        return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def parse_file_meta(path: Path, sf_override: Optional[int], preamble_override: Optional[int]) -> FileMeta:
    match = FILENAME_RE.match(path.stem)
    if not match:
        if sf_override is None or preamble_override is None:
            raise ValueError(
                f"Cannot parse metadata from {path.name}. "
                "Use --sf and --preamble-len for nonstandard names."
            )
        return FileMeta(
            file_name=path.name,
            experiment_id="",
            corridor_id="",
            position_id=path.stem,
            sf=sf_override,
            tx_power_dbm=0,
            preamble_len=preamble_override,
        )

    groups = match.groupdict()
    return FileMeta(
        file_name=path.name,
        experiment_id=groups["experiment"],
        corridor_id=groups["corridor"],
        position_id=groups["position"],
        sf=sf_override if sf_override is not None else int(groups["sf"]),
        tx_power_dbm=int(groups["tx_power"]),
        preamble_len=preamble_override if preamble_override is not None else int(groups["preamble"]),
    )


def build_runtime(sf: int, sample_rate: float, bandwidth: float) -> LoraRuntime:
    os_float = sample_rate / bandwidth
    os_factor = int(round(os_float))
    if abs(os_float - os_factor) > 1e-6:
        raise ValueError(
            f"sample_rate / bandwidth must be an integer for this script. "
            f"Got {sample_rate} / {bandwidth} = {os_float:.6f}."
        )

    fft_size = 1 << sf
    n = np.arange(fft_size, dtype=np.float64)
    upchirp = np.exp(1j * np.pi * n * (n / fft_size - 1.0))
    downchirp = np.conj(upchirp).astype(np.complex64)
    return LoraRuntime(
        sf=sf,
        sample_rate=sample_rate,
        bandwidth=bandwidth,
        os_factor=os_factor,
        fft_size=fft_size,
        symbol_samples=fft_size * os_factor,
        downchirp=downchirp,
    )


def downsample_symbol(symbol: np.ndarray, runtime: LoraRuntime, method: str) -> np.ndarray:
    symbol = symbol[: runtime.symbol_samples]
    if method == "pick":
        return symbol[:: runtime.os_factor][: runtime.fft_size]
    return symbol.reshape(runtime.fft_size, runtime.os_factor).mean(axis=1)


def dechirp_fft(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    downsample_method: str,
    remove_dc: bool,
) -> Optional[np.ndarray]:
    symbol = iq.read(start, runtime.symbol_samples)
    if symbol is None or symbol.size != runtime.symbol_samples:
        return None

    baseband = downsample_symbol(symbol, runtime, downsample_method)
    if remove_dc:
        baseband = baseband - np.mean(baseband)
    return np.fft.fft(baseband * runtime.downchirp, n=runtime.fft_size)


def peak_width_bins(mag: np.ndarray, peak_bin: int, threshold_db: float) -> float:
    threshold = mag[peak_bin] * (10.0 ** (threshold_db / 20.0))
    width = 1
    n = mag.size

    for step in range(1, n // 2):
        if mag[(peak_bin - step) % n] < threshold:
            break
        width += 1

    for step in range(1, n // 2):
        if mag[(peak_bin + step) % n] < threshold:
            break
        width += 1

    return float(width)


def power_spectrum_metrics(power: np.ndarray, peak_width_db: float) -> tuple[int, float, float]:
    peak_bin = int(np.argmax(power))
    peak_power = float(power[peak_bin])
    residual_power = float(np.sum(power) - peak_power)
    peak_to_residual_db = 10.0 * math.log10((peak_power + EPS) / (residual_power + EPS))
    width = peak_width_bins(np.sqrt(power), peak_bin, peak_width_db)
    return peak_bin, peak_to_residual_db, width


def spectrum_metrics(spec: np.ndarray, peak_width_db: float) -> tuple[int, float, float]:
    power = np.abs(spec) ** 2
    return power_spectrum_metrics(power, peak_width_db)


def circular_distance_bins(a: float, b: float, n_bins: int) -> float:
    distance = abs(a - b) % n_bins
    return float(min(distance, n_bins - distance))


def circular_mean_bin(peaks: Sequence[int], n_bins: int) -> float:
    if not peaks:
        return float("nan")
    angles = 2.0 * np.pi * np.asarray(peaks, dtype=np.float64) / n_bins
    mean_vector = np.mean(np.exp(1j * angles))
    angle = float(np.angle(mean_vector))
    if angle < 0:
        angle += 2.0 * np.pi
    return angle * n_bins / (2.0 * np.pi)


def circular_std_bins(peaks: Sequence[int], n_bins: int) -> float:
    if len(peaks) <= 1:
        return 0.0
    angles = 2.0 * np.pi * np.asarray(peaks, dtype=np.float64) / n_bins
    resultant = abs(np.mean(np.exp(1j * angles)))
    resultant = min(max(resultant, EPS), 1.0)
    std_angle = math.sqrt(-2.0 * math.log(resultant))
    return float(std_angle * n_bins / (2.0 * np.pi))


def evaluate_candidate(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    symbol_count: int,
    downsample_method: str,
    remove_dc: bool,
    peak_width_db: float,
) -> Optional[Candidate]:
    scores: List[float] = []
    peaks: List[int] = []
    widths: List[float] = []

    for symbol_idx in range(symbol_count):
        spec = dechirp_fft(
            iq,
            start + symbol_idx * runtime.symbol_samples,
            runtime,
            downsample_method,
            remove_dc,
        )
        if spec is None:
            return None

        peak_bin, score_db, width_bins = spectrum_metrics(spec, peak_width_db)
        scores.append(score_db)
        peaks.append(peak_bin)
        widths.append(width_bins)

    return Candidate(
        sample_start=start,
        score_db=float(np.mean(scores)),
        peak_bin_mean=circular_mean_bin(peaks, runtime.fft_size),
        peak_bin_std=circular_std_bins(peaks, runtime.fft_size),
        peak_width_bins_avg=float(np.mean(widths)),
        symbol_scores_db=scores,
        symbol_peak_bins=peaks,
    )


def evaluate_candidate_noncoherent(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    symbol_count: int,
    downsample_method: str,
    remove_dc: bool,
    peak_width_db: float,
) -> Optional[Candidate]:
    powers: List[np.ndarray] = []
    peaks: List[int] = []
    widths: List[float] = []
    scores: List[float] = []

    for symbol_idx in range(symbol_count):
        spec = dechirp_fft(
            iq,
            start + symbol_idx * runtime.symbol_samples,
            runtime,
            downsample_method,
            remove_dc,
        )
        if spec is None:
            return None
        power = np.abs(spec) ** 2
        peak_bin, score_db, width_bins = power_spectrum_metrics(power, peak_width_db)
        powers.append(power)
        peaks.append(peak_bin)
        scores.append(score_db)
        widths.append(width_bins)

    summed_power = np.sum(np.vstack(powers), axis=0)
    common_peak, common_score_db, common_width = power_spectrum_metrics(summed_power, peak_width_db)
    return Candidate(
        sample_start=start,
        score_db=float(common_score_db),
        peak_bin_mean=float(common_peak),
        peak_bin_std=circular_std_bins(peaks, runtime.fft_size),
        peak_width_bins_avg=float(common_width),
        symbol_scores_db=scores,
        symbol_peak_bins=peaks,
    )


def evaluate_candidate_for_detection(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    symbol_count: int,
    downsample_method: str,
    remove_dc: bool,
    peak_width_db: float,
    detection_mode: str,
) -> Optional[Candidate]:
    if detection_mode == "noncoherent":
        return evaluate_candidate_noncoherent(
            iq,
            start,
            runtime,
            symbol_count,
            downsample_method,
            remove_dc,
            peak_width_db,
        )
    return evaluate_candidate(
        iq,
        start,
        runtime,
        symbol_count,
        downsample_method,
        remove_dc,
        peak_width_db,
    )


def candidate_passes(
    candidate: Candidate,
    threshold_db: float,
    peak_std_max: float,
    require_peak_std: bool = True,
) -> bool:
    if candidate.score_db < threshold_db:
        return False
    return (not require_peak_std) or candidate.peak_bin_std <= peak_std_max


def refine_candidate(
    iq: IQMemmap,
    coarse_start: int,
    scan_step: int,
    runtime: LoraRuntime,
    detect_symbols: int,
    downsample_method: str,
    remove_dc: bool,
    peak_width_db: float,
    refine_step: int,
    detection_mode: str = "per-symbol",
) -> Optional[Candidate]:
    low = max(0, coarse_start - scan_step)
    high = min(
        iq.num_complex - detect_symbols * runtime.symbol_samples,
        coarse_start + scan_step,
    )
    best: Optional[Candidate] = None

    for start in range(low, high + 1, refine_step):
        candidate = evaluate_candidate_for_detection(
            iq,
            start,
            runtime,
            detect_symbols,
            downsample_method,
            remove_dc,
            peak_width_db,
            detection_mode,
        )
        if candidate is None:
            continue
        if best is None or candidate.score_db > best.score_db:
            best = candidate

    return best


def rewind_to_first_preamble_symbol(
    iq: IQMemmap,
    candidate: Candidate,
    runtime: LoraRuntime,
    preamble_len: int,
    downsample_method: str,
    remove_dc: bool,
    peak_width_db: float,
    threshold_db: float,
    peak_bin_tolerance: float,
) -> Candidate:
    start = candidate.sample_start
    reference_peak = candidate.peak_bin_mean

    for _ in range(max(0, preamble_len - 1)):
        previous = start - runtime.symbol_samples
        if previous < 0:
            break

        previous_candidate = evaluate_candidate(
            iq,
            previous,
            runtime,
            1,
            downsample_method,
            remove_dc,
            peak_width_db,
        )
        if previous_candidate is None or previous_candidate.score_db < threshold_db:
            break
        if circular_distance_bins(previous_candidate.peak_bin_mean, reference_peak, runtime.fft_size) > peak_bin_tolerance:
            break

        start = previous

    rewound = evaluate_candidate(
        iq,
        start,
        runtime,
        len(candidate.symbol_scores_db),
        downsample_method,
        remove_dc,
        peak_width_db,
    )
    return rewound if rewound is not None else candidate


def detect_packets(
    iq: IQMemmap,
    runtime: LoraRuntime,
    preamble_len: int,
    args: argparse.Namespace,
) -> List[Candidate]:
    scan_step = max(1, int(round(args.scan_step_symbols * runtime.symbol_samples)))
    refine_step = max(1, args.refine_step_samples or runtime.os_factor)
    suppress_samples = max(
        runtime.symbol_samples,
        int(round(args.suppress_symbols * runtime.symbol_samples)),
    )
    detect_symbols = min(args.detect_symbols, preamble_len)
    max_scan_start = iq.num_complex - detect_symbols * runtime.symbol_samples
    if args.max_scan_symbols is not None:
        max_scan_start = min(max_scan_start, args.max_scan_symbols * runtime.symbol_samples)

    packets: List[Candidate] = []
    start = 0
    checked = 0

    while start <= max_scan_start:
        candidate = evaluate_candidate_for_detection(
            iq,
            start,
            runtime,
            detect_symbols,
            args.downsample,
            args.remove_dc,
            args.peak_width_db,
            args.detection_mode,
        )
        checked += 1

        if candidate is not None and candidate_passes(
            candidate,
            args.detect_threshold_db,
            args.peak_std_max,
            args.detection_mode != "noncoherent",
        ):
            refined = refine_candidate(
                iq,
                start,
                scan_step,
                runtime,
                detect_symbols,
                args.downsample,
                args.remove_dc,
                args.peak_width_db,
                refine_step,
                args.detection_mode,
            )
            if refined is not None:
                candidate = rewind_to_first_preamble_symbol(
                    iq,
                    refined,
                    runtime,
                    preamble_len,
                    args.downsample,
                    args.remove_dc,
                    args.peak_width_db,
                    args.detect_threshold_db,
                    args.peak_bin_tolerance,
                )

            if candidate_passes(
                candidate,
                args.detect_threshold_db,
                args.peak_std_max,
                args.detection_mode != "noncoherent",
            ):
                if packets and candidate.sample_start - packets[-1].sample_start < suppress_samples:
                    if candidate.score_db > packets[-1].score_db:
                        packets[-1] = candidate
                else:
                    packets.append(candidate)

                if args.max_packets_per_file and len(packets) >= args.max_packets_per_file:
                    break

                start = max(start + scan_step, candidate.sample_start + suppress_samples)
                continue

        if args.verbose and checked % args.progress_every == 0:
            pct = 100.0 * start / max(max_scan_start, 1)
            print(f"  scan {pct:5.1f}%: checked={checked}, packets={len(packets)}")

        start += scan_step

    return packets


def make_bin_offsets(bin_count: int) -> List[int]:
    if bin_count <= 0:
        raise ValueError("--bin-count must be positive")
    left = bin_count // 2
    return list(range(-left, bin_count - left))


def offset_suffix(offset: int) -> str:
    return f"{offset:+d}"


def power_db(samples: np.ndarray) -> float:
    return 10.0 * math.log10(float(np.mean(np.abs(samples) ** 2)) + EPS)


def mag_to_db(mag: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    values = 20.0 * np.log10(np.maximum(mag, EPS))
    return np.maximum(values, floor_db)


def stft_matrix(samples: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    if samples.size < nfft:
        return np.empty((0, nfft), dtype=np.complex64)

    window = np.hanning(nfft).astype(np.float32)
    frames = []
    for start in range(0, samples.size - nfft + 1, hop):
        frame = samples[start : start + nfft]
        frames.append(np.fft.fftshift(np.fft.fft(frame * window, n=nfft)))
    return np.vstack(frames)


def relative_power_db_from_complex(values: np.ndarray, floor_db: float, ref_percentile: float = 99.7) -> np.ndarray:
    power = np.abs(values) ** 2
    ref = np.percentile(power, ref_percentile)
    if ref <= EPS:
        ref = float(np.max(power))
    db = 10.0 * np.log10((power + EPS) / (ref + EPS))
    return np.maximum(db, floor_db)


def signed_bin_axis(n_bins: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / n_bins)).astype(int)


def signed_power_profile(spec_power: np.ndarray, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    n_bins = spec_power.size
    signed_bins = signed_bin_axis(n_bins)
    shifted = np.fft.fftshift(spec_power)
    keep = (signed_bins >= -half_width) & (signed_bins <= half_width)
    return signed_bins[keep], shifted[keep]


def line_power_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10((power + EPS) / (float(np.max(power)) + EPS))


def collect_preamble_spectra(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    symbol_count: int,
    args: argparse.Namespace,
) -> tuple[List[np.ndarray], List[int]]:
    spectra: List[np.ndarray] = []
    peak_bins: List[int] = []
    for symbol_idx in range(symbol_count):
        spec = dechirp_fft(
            iq,
            start + symbol_idx * runtime.symbol_samples,
            runtime,
            args.downsample,
            args.remove_dc,
        )
        if spec is None:
            break
        peak_bin, _, _ = spectrum_metrics(spec, args.peak_width_db)
        spectra.append(spec)
        peak_bins.append(peak_bin)
    return spectra, peak_bins


def plot_matplotlib_diagnostics(
    iq: IQMemmap,
    candidate: Candidate,
    runtime: LoraRuntime,
    meta: FileMeta,
    packet_index: int,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    event_symbols = args.plot_event_symbols
    if event_symbols is None:
        event_symbols = meta.preamble_len + 6.0
    event_samples = int(round(event_symbols * runtime.symbol_samples))
    event_samples = min(event_samples, iq.num_complex - candidate.sample_start)
    segment = iq.read(candidate.sample_start, event_samples)
    if segment is None:
        print(f"  skip plots for packet {packet_index}: segment out of range")
        return
    if args.remove_dc:
        segment = segment - np.mean(segment)

    prefix = f"{Path(meta.file_name).stem}_pkt{packet_index:03d}_s{candidate.sample_start}_matplotlib"

    stft = stft_matrix(segment, args.stft_nfft, args.stft_hop)
    if stft.size:
        stft_db = relative_power_db_from_complex(stft, args.stft_floor_db, args.stft_ref_percentile)
        extent = [
            0.0,
            segment.size / runtime.sample_rate * 1e3,
            -runtime.sample_rate / 2.0 / 1e3,
            runtime.sample_rate / 2.0 / 1e3,
        ]

        fig, ax = plt.subplots(figsize=(13.5, 4.4), constrained_layout=True)
        im = ax.imshow(
            stft_db.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=args.matplotlib_cmap,
            vmin=args.stft_floor_db,
            vmax=0.0,
            interpolation="bilinear",
        )

        symbol_ms = runtime.symbol_samples / runtime.sample_rate * 1e3
        for symbol_idx in range(1, int(math.floor(event_symbols)) + 1):
            ax.axvline(symbol_idx * symbol_ms, color="white", alpha=0.14, linewidth=0.8)

        markers = [
            ("sync", meta.preamble_len * symbol_ms),
            ("SFD", (meta.preamble_len + 2.0) * symbol_ms),
            ("quarter", (meta.preamble_len + 4.25) * symbol_ms),
        ]
        y_top = runtime.sample_rate / 2.0 / 1e3
        for label, x_ms in markers:
            if 0.0 <= x_ms <= extent[1]:
                ax.axvline(x_ms, color="white", linewidth=1.15, alpha=0.95)
                ax.text(
                    x_ms + 0.15,
                    y_top - 18,
                    label,
                    color="white",
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=10,
                )

        ax.set_title(f"Located LoRa preamble + sync + SFD (packet event {packet_index}, valid=1)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=ax, label="Relative power (dB)")
        fig.savefig(args.plot_dir / f"{prefix}_located_stft.png", dpi=args.plot_dpi)
        plt.close(fig)

    profile_start_symbol = args.plot_start_symbol
    if profile_start_symbol is None:
        profile_start_symbol = args.skip_preamble_symbols
    profile_start_symbol = max(0, min(profile_start_symbol, meta.preamble_len - 1))
    profile_symbols = args.plot_symbols if args.plot_symbols is not None else min(meta.preamble_len - profile_start_symbol, 8)
    profile_symbols = max(1, min(profile_symbols, meta.preamble_len - profile_start_symbol))
    profile_start = candidate.sample_start + profile_start_symbol * runtime.symbol_samples
    spectra, peak_bins = collect_preamble_spectra(iq, profile_start, runtime, profile_symbols, args)
    if not spectra:
        print(f"  skip DFT profile for packet {packet_index}: no spectra extracted")
        return

    spectra_arr = np.vstack(spectra)
    noncoherent_before = np.mean(np.abs(spectra_arr) ** 2, axis=0)
    before_peak_bin = int(np.argmax(noncoherent_before))
    before_signed_peak = before_peak_bin if before_peak_bin <= runtime.fft_size // 2 else before_peak_bin - runtime.fft_size
    x_before, y_before_power = signed_power_profile(noncoherent_before, args.plot_profile_bins)
    y_before = line_power_db(y_before_power)

    aligned_spectra = []
    for spec, peak_bin in zip(spectra, peak_bins):
        aligned = np.roll(spec, -peak_bin)
        aligned *= np.exp(-1j * np.angle(aligned[0]))
        aligned_spectra.append(aligned)
    aligned_arr = np.vstack(aligned_spectra)
    coherent_after = np.abs(np.mean(aligned_arr, axis=0)) ** 2
    noncoherent_after = np.mean(np.abs(aligned_arr) ** 2, axis=0)
    x_after, y_coh_power = signed_power_profile(coherent_after, args.plot_profile_bins)
    _, y_noncoh_power = signed_power_profile(noncoherent_after, args.plot_profile_bins)
    y_coh = line_power_db(y_coh_power)
    y_noncoh = line_power_db(y_noncoh_power)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.5), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Packet {packet_index:03d} preamble DFT validation | "
        f"start={candidate.sample_start} sample, peak={before_signed_peak} bin",
        fontsize=12,
    )

    axes[0].plot(x_before, y_before, color="#1f77b4", linewidth=1.25)
    axes[0].axvline(before_signed_peak, color="red", linestyle=":", linewidth=1.0)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.75)
    axes[0].set_title(f"Before peak alignment: noncoherent sum, peak bin {before_signed_peak}")
    axes[0].set_ylabel("Relative power (dB)")
    axes[0].grid(True, alpha=0.28)
    axes[0].set_ylim(args.profile_floor_db, 2)

    axes[1].plot(x_after, y_coh, color="#2ca02c", linewidth=1.25, label="coherent")
    axes[1].plot(x_after, y_noncoh, color="#ff7f0e", linewidth=1.1, label="noncoherent")
    axes[1].axvline(0, color="red", linestyle=":", linewidth=1.0)
    axes[1].set_title("After peak alignment: coherent peak bin 0")
    axes[1].set_xlabel("Signed DFT bin")
    axes[1].set_ylabel("Relative power (dB)")
    axes[1].grid(True, alpha=0.28)
    axes[1].set_ylim(args.profile_floor_db, 2)
    axes[1].legend(loc="lower right")

    fig.savefig(args.plot_dir / f"{prefix}_dft_profile.png", dpi=args.plot_dpi)
    plt.close(fig)


def plot_packet_diagnostics(
    iq: IQMemmap,
    candidate: Candidate,
    runtime: LoraRuntime,
    meta: FileMeta,
    packet_index: int,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_style == "matplotlib":
        plot_matplotlib_diagnostics(iq, candidate, runtime, meta, packet_index, args)
        return

    is_paper_style = args.plot_style == "paper"
    plot_start_symbol = args.plot_start_symbol
    if plot_start_symbol is None:
        plot_start_symbol = args.skip_preamble_symbols if is_paper_style else 0
    plot_start_symbol = max(0, min(plot_start_symbol, meta.preamble_len - 1))

    plot_symbols = args.plot_symbols if args.plot_symbols is not None else min(meta.preamble_len, 16)
    plot_symbols = max(1, min(plot_symbols, meta.preamble_len - plot_start_symbol))
    plot_sample_start = candidate.sample_start + plot_start_symbol * runtime.symbol_samples
    sample_count = plot_symbols * runtime.symbol_samples
    segment = iq.read(plot_sample_start, sample_count)
    if segment is None:
        print(f"  skip plots for packet {packet_index}: segment out of range")
        return

    if args.remove_dc:
        segment = segment - np.mean(segment)

    prefix = (
        f"{Path(meta.file_name).stem}"
        f"_pkt{packet_index:03d}"
        f"_s{candidate.sample_start}"
        f"_{args.plot_style}"
    )

    stft = stft_matrix(segment, args.stft_nfft, args.stft_hop)
    if stft.size:
        freqs = np.fft.fftshift(np.fft.fftfreq(args.stft_nfft, d=1.0 / runtime.sample_rate))
        if is_paper_style:
            keep = np.abs(freqs) <= (runtime.bandwidth / 2.0)
            stft = stft[:, keep]
            freqs = freqs[keep]

        stft_db = mag_to_db(np.abs(stft), floor_db=-200.0)
        stft_db = stft_db - np.max(stft_db)
        stft_db = np.maximum(stft_db, args.stft_floor_db)

        if is_paper_style:
            x_max = segment.size / runtime.symbol_samples
            extent = [0.0, x_max, freqs[0] / 1e3, freqs[-1] / 1e3]
            fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
            cmap = args.paper_cmap
            x_label = "Time (LoRa symbols)"
            y_label = "Frequency offset (kHz)"
            title = f"LoRa preamble spectrogram, SF{meta.sf}, BW={runtime.bandwidth / 1e3:.0f} kHz"
        else:
            extent = [
                0.0,
                segment.size / runtime.sample_rate * 1e3,
                freqs[0] / 1e3,
                freqs[-1] / 1e3,
            ]
            fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
            cmap = "magma"
            x_label = "time after detected preamble start (ms)"
            y_label = "frequency (kHz)"
            title = f"{meta.file_name} packet {packet_index}: synced preamble STFT"

        im = ax.imshow(
            stft_db.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=args.stft_floor_db,
            vmax=0.0,
            interpolation="bilinear" if is_paper_style else "nearest",
        )
        if not is_paper_style:
            for idx in range(plot_symbols + 1):
                x_ms = idx * runtime.symbol_samples / runtime.sample_rate * 1e3
                ax.axvline(x_ms, color="white", alpha=0.18, linewidth=0.7)
        else:
            ax.set_ylim(-runtime.bandwidth / 2.0 / 1e3, runtime.bandwidth / 2.0 / 1e3)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, label="relative magnitude (dB)")
        fig.savefig(args.plot_dir / f"{prefix}_stft.png", dpi=args.plot_dpi)
        plt.close(fig)

    spectra = []
    peak_bins = []
    for symbol_idx in range(plot_symbols):
        spec = dechirp_fft(
            iq,
            plot_sample_start + symbol_idx * runtime.symbol_samples,
            runtime,
            args.downsample,
            args.remove_dc,
        )
        if spec is None:
            break
        peak_bin, _, _ = spectrum_metrics(spec, args.peak_width_db)
        spectra.append(spec)
        peak_bins.append(peak_bin)

    if not spectra:
        print(f"  skip plots for packet {packet_index}: no spectra extracted")
        return

    mag = np.vstack([np.abs(spec) for spec in spectra])
    mag_rel = mag / (np.max(mag, axis=1, keepdims=True) + EPS)
    mag_db = mag_to_db(mag_rel, floor_db=args.stft_floor_db)

    if is_paper_style:
        fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
        fft_cmap = args.paper_cmap
        interp = "nearest"
        title = "Downchirped preamble DFT magnitude"
    else:
        fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
        fft_cmap = "viridis"
        interp = "nearest"
        title = f"{meta.file_name} packet {packet_index}: dechirp + FFT"
    im = ax.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        interpolation=interp,
        cmap=fft_cmap,
        extent=[0, runtime.fft_size - 1, -0.5, len(spectra) - 0.5],
        vmin=args.stft_floor_db,
        vmax=0.0,
    )
    ax.plot(peak_bins, np.arange(len(peak_bins)), ".", color="#d62728", markersize=4, label="max DFT bin")
    ax.set_title(title)
    ax.set_xlabel("DFT bin before peak alignment")
    ax.set_ylabel("preamble symbol index")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="relative magnitude (dB)")
    fig.savefig(args.plot_dir / f"{prefix}_dechirp_fft_full.png", dpi=args.plot_dpi)
    plt.close(fig)

    local_offsets = make_bin_offsets(args.plot_bin_count)
    local_indices = np.mod(np.asarray(local_offsets, dtype=np.int64), runtime.fft_size)
    local_rows = []
    for spec, peak_bin in zip(spectra, peak_bins):
        aligned = np.roll(spec, -peak_bin)
        local = np.abs(aligned[local_indices])
        local = local / (np.max(local) + EPS)
        local_rows.append(local)
    local_db = mag_to_db(np.vstack(local_rows), floor_db=args.stft_floor_db)

    if is_paper_style:
        fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
        local_cmap = args.paper_cmap
        title = "Peak-aligned local DFT magnitude"
    else:
        fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        local_cmap = "viridis"
        title = f"{meta.file_name} packet {packet_index}: peak-aligned local FFT"
    im = ax.imshow(
        local_db,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=local_cmap,
        extent=[local_offsets[0] - 0.5, local_offsets[-1] + 0.5, -0.5, len(local_rows) - 0.5],
        vmin=args.stft_floor_db,
        vmax=0.0,
    )
    ax.axvline(0, color="red", linewidth=1.0, alpha=0.75, label="aligned bin0")
    ax.set_title(title)
    ax.set_xlabel("DFT bin offset after peak alignment")
    ax.set_ylabel("preamble symbol index")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="relative magnitude (dB)")
    fig.savefig(args.plot_dir / f"{prefix}_dechirp_fft_aligned_local.png", dpi=args.plot_dpi)
    plt.close(fig)


def extract_packet_features(
    iq: IQMemmap,
    candidate: Candidate,
    runtime: LoraRuntime,
    meta: FileMeta,
    packet_index: int,
    offsets: Sequence[int],
    args: argparse.Namespace,
) -> dict:
    # 1) 选择用于生成指纹的前导码 symbol。
    #    LoRa 前导码前几个 symbol 有时受同步搜索影响更大，所以可以用
    #    --skip-preamble-symbols 跳过开头；--feature-symbols 控制后续平均多少个 symbol。
    skip_symbols = max(0, min(args.skip_preamble_symbols, meta.preamble_len - 1))
    available_symbols = meta.preamble_len - skip_symbols
    symbol_count = args.feature_symbols if args.feature_symbols is not None else available_symbols
    symbol_count = max(1, min(symbol_count, available_symbols))

    spectra = []
    peak_bins = []
    scores = []
    widths = []
    local_complex = []
    local_mag = []
    bin_indices = np.mod(np.asarray(offsets, dtype=np.int64), runtime.fft_size)

    # 2) 对每个前导码 symbol 做 dechirp + FFT。
    #    dechirp_fft() 内部会读取一个 LoRa symbol 长度的 IQ，
    #    乘以下啁啾 downchirp，然后做 FFT。理想情况下，前导码上啁啾
    #    被 dechirp 后会变成一个很尖的频域峰。
    for symbol_idx in range(skip_symbols, skip_symbols + symbol_count):
        spec = dechirp_fft(
            iq,
            candidate.sample_start + symbol_idx * runtime.symbol_samples,
            runtime,
            args.downsample,
            args.remove_dc,
        )
        if spec is None:
            break

        spectra.append(spec)
        peak_bin, score_db, width_bins = spectrum_metrics(spec, args.peak_width_db)
        peak_bins.append(peak_bin)
        scores.append(score_db)
        widths.append(width_bins)

    if not spectra:
        raise RuntimeError(f"Could not extract features at sample {candidate.sample_start}")

    # 3) 确定每个 symbol 的频域峰位置，并把这个峰平移到 bin0。
    #    per-symbol: 每个 symbol 用自己的 peak_bin 对齐。
    #    noncoherent: 多个 symbol 的功率谱先非相干累加，再用共同峰值对齐。
    if args.feature_alignment == "noncoherent":
        summed_power = np.sum(np.vstack([np.abs(spec) ** 2 for spec in spectra]), axis=0)
        align_bins = [int(np.argmax(summed_power))] * len(spectra)
    else:
        align_bins = peak_bins

    # 4) 对齐后只取 bin0 附近的局部频谱。
    #    默认 bin_count=16 时 offsets 是 [-8, ..., +7]，
    #    也就是取 16 个 FFT bin。每个 bin 保留复数值，后面拆成幅度和相位。
    for spec, align_bin in zip(spectra, align_bins):
        aligned = np.roll(spec, -align_bin)
        local = aligned[bin_indices].astype(np.complex128)
        center = aligned[0]

        # 相位默认做相对化：把中心 bin 的相位旋转到 0，
        # 这样减少包初相带来的随机性；normalize=peak 会把中心峰幅度归一化。
        if args.phase_mode in ("relative", "both"):
            local *= np.exp(-1j * np.angle(center))
        if args.normalize == "peak":
            local /= abs(center) + EPS
        elif args.normalize == "energy":
            local /= math.sqrt(float(np.sum(np.abs(spec) ** 2)) + EPS)

        local_complex.append(local)
        local_mag.append(np.abs(local))

    local_complex_arr = np.vstack(local_complex)
    local_mag_arr = np.vstack(local_mag)

    # 5) 多个前导码 symbol 做平均，得到一个包级别的指纹。
    #    幅度直接平均；相位先平均复数，再取 angle，避免直接平均角度导致跳变问题。
    complex_mean = np.mean(local_complex_arr, axis=0)
    mag_mean = np.mean(local_mag_arr, axis=0)
    phase_mean = np.angle(complex_mean)

    power_symbols = args.power_symbols if args.power_symbols is not None else symbol_count
    power_start = candidate.sample_start + skip_symbols * runtime.symbol_samples
    power_samples = min(iq.num_complex - power_start, power_symbols * runtime.symbol_samples)
    preamble_samples = iq.read(power_start, power_samples)
    preamble_power_db = power_db(preamble_samples) if preamble_samples is not None else float("nan")

    row = {
        "file_name": meta.file_name,
        "experiment_id": meta.experiment_id,
        "corridor_id": meta.corridor_id,
        "position_id": meta.position_id,
        "filename_sf": meta.sf,
        "filename_tx_power_dbm": meta.tx_power_dbm,
        "filename_preamble_len": meta.preamble_len,
        "packet_index": packet_index,
        "sample_start": candidate.sample_start,
        "sample_rate": runtime.sample_rate,
        "bandwidth": runtime.bandwidth,
        "fft_size": runtime.fft_size,
        "symbol_samples": runtime.symbol_samples,
        "skip_preamble_symbols": skip_symbols,
        "feature_symbols": len(local_complex),
        "feature_bin_count": len(offsets),
        "normalize": args.normalize,
        "phase_mode": args.phase_mode,
        "preamble_avg_power_db": preamble_power_db,
        "preamble_peak_to_residual_db": float(np.mean(scores)),
        "preamble_peak_width_3db_bins_avg": float(np.mean(widths)),
        "preamble_peak_bin_mean": circular_mean_bin(peak_bins, runtime.fft_size),
        "preamble_peak_bin_std": circular_std_bins(peak_bins, runtime.fft_size),
        "detect_score_db": candidate.score_db,
        "detect_peak_bin_mean": candidate.peak_bin_mean,
        "detect_peak_bin_std": candidate.peak_bin_std,
    }

    if args.phase_mode == "both":
        raw_complex_rows = []
        for symbol_idx in range(len(local_complex)):
            spec = spectra[symbol_idx]
            peak_bin = peak_bins[symbol_idx]
            aligned = np.roll(spec, -peak_bin)
            raw_local = aligned[bin_indices].astype(np.complex128)
            if args.normalize == "peak":
                raw_local /= abs(aligned[0]) + EPS
            elif args.normalize == "energy":
                raw_local /= math.sqrt(float(np.sum(np.abs(spec) ** 2)) + EPS)
            raw_complex_rows.append(raw_local)
        raw_phase_mean = np.angle(np.mean(np.vstack(raw_complex_rows), axis=0))
    else:
        raw_phase_mean = None

    for idx, offset in enumerate(offsets):
        suffix = offset_suffix(offset)
        row[f"preamble_fft_mag_bin_{suffix}"] = float(mag_mean[idx])
    for idx, offset in enumerate(offsets):
        suffix = offset_suffix(offset)
        row[f"preamble_fft_phase_bin_{suffix}"] = float(phase_mean[idx])
    if raw_phase_mean is not None:
        for idx, offset in enumerate(offsets):
            suffix = offset_suffix(offset)
            row[f"preamble_fft_phase_raw_bin_{suffix}"] = float(raw_phase_mean[idx])

    # 最终 CSV 每一行对应一个 LoRa 包：
    # - preamble_fft_mag_bin_-8 ... +7 是 16 个局部 bin 的幅度
    # - preamble_fft_phase_bin_-8 ... +7 是对应 16 个 bin 的相位
    return row


def feature_row_failure(row: dict, args: argparse.Namespace) -> Optional[str]:
    peak_std = float(row["preamble_peak_bin_std"])
    score_db = float(row["preamble_peak_to_residual_db"])
    if args.verify_peak_std and (not math.isfinite(peak_std) or peak_std > args.verify_peak_std_max):
        return f"peak std {peak_std:.3f} > {args.verify_peak_std_max:.3f}"
    if not math.isfinite(score_db) or score_db < args.verify_score_db:
        return f"score {score_db:.3f} dB < {args.verify_score_db:.3f} dB"
    return None


def extract_valid_packet_features(
    iq: IQMemmap,
    candidate: Candidate,
    runtime: LoraRuntime,
    meta: FileMeta,
    packet_index: int,
    offsets: Sequence[int],
    args: argparse.Namespace,
) -> Optional[dict]:
    try:
        row = extract_packet_features(iq, candidate, runtime, meta, packet_index, offsets, args)
    except RuntimeError as exc:
        if args.verbose:
            print(f"  skip candidate at {candidate.sample_start}: {exc}")
        return None

    failure = feature_row_failure(row, args)
    if failure is not None:
        if args.verbose:
            print(f"  reject candidate at {candidate.sample_start}: {failure}")
        return None
    return row


def refine_candidate_window(
    iq: IQMemmap,
    center_start: int,
    window_samples: int,
    runtime: LoraRuntime,
    detect_symbols: int,
    args: argparse.Namespace,
) -> Optional[Candidate]:
    refine_step = max(1, args.refine_step_samples or runtime.os_factor)
    max_start = iq.num_complex - detect_symbols * runtime.symbol_samples
    low = max(0, center_start - window_samples)
    high = min(max_start, center_start + window_samples)
    best: Optional[Candidate] = None

    for start in range(low, high + 1, refine_step):
        candidate = evaluate_candidate_for_detection(
            iq,
            start,
            runtime,
            detect_symbols,
            args.downsample,
            args.remove_dc,
            args.peak_width_db,
            args.detection_mode,
        )
        if candidate is None:
            continue
        if best is None or candidate.score_db > best.score_db:
            best = candidate
    return best


def find_seed_packet(
    iq: IQMemmap,
    runtime: LoraRuntime,
    meta: FileMeta,
    offsets: Sequence[int],
    args: argparse.Namespace,
) -> tuple[Optional[Candidate], Optional[dict], int]:
    scan_step = max(1, int(round(args.scan_step_symbols * runtime.symbol_samples)))
    refine_step = max(1, args.refine_step_samples or runtime.os_factor)
    suppress_samples = max(
        runtime.symbol_samples,
        int(round(args.suppress_symbols * runtime.symbol_samples)),
    )
    detect_symbols = min(args.detect_symbols, meta.preamble_len)
    max_scan_start = iq.num_complex - detect_symbols * runtime.symbol_samples
    if args.seed_max_scan_symbols is not None:
        max_scan_start = min(max_scan_start, args.seed_max_scan_symbols * runtime.symbol_samples)
    elif args.max_scan_symbols is not None:
        max_scan_start = min(max_scan_start, args.max_scan_symbols * runtime.symbol_samples)

    start = 0
    checked = 0
    while start <= max_scan_start:
        candidate = evaluate_candidate_for_detection(
            iq,
            start,
            runtime,
            detect_symbols,
            args.downsample,
            args.remove_dc,
            args.peak_width_db,
            args.detection_mode,
        )
        checked += 1

        if candidate is not None and candidate_passes(
            candidate,
            args.detect_threshold_db,
            args.peak_std_max,
            args.detection_mode != "noncoherent",
        ):
            refined = refine_candidate(
                iq,
                start,
                scan_step,
                runtime,
                detect_symbols,
                args.downsample,
                args.remove_dc,
                args.peak_width_db,
                refine_step,
                args.detection_mode,
            )
            if refined is not None:
                candidate = rewind_to_first_preamble_symbol(
                    iq,
                    refined,
                    runtime,
                    meta.preamble_len,
                    args.downsample,
                    args.remove_dc,
                    args.peak_width_db,
                    args.detect_threshold_db,
                    args.peak_bin_tolerance,
                )

            if candidate_passes(
                candidate,
                args.detect_threshold_db,
                args.peak_std_max,
                args.detection_mode != "noncoherent",
            ):
                row = extract_valid_packet_features(
                    iq,
                    candidate,
                    runtime,
                    meta,
                    0,
                    offsets,
                    args,
                )
                if row is not None:
                    return candidate, row, checked
                start = max(start + scan_step, candidate.sample_start + suppress_samples)
                continue

        if args.verbose and checked % args.progress_every == 0:
            pct = 100.0 * start / max(max_scan_start, 1)
            print(f"  seed scan {pct:5.1f}%: checked={checked}")
        start += scan_step

    return None, None, checked


def process_file_periodic(path: Path, args: argparse.Namespace, offsets: Sequence[int]) -> List[dict]:
    meta = parse_file_meta(path, args.sf, args.preamble_len)
    runtime = build_runtime(meta.sf, args.sample_rate, args.bandwidth)
    iq = IQMemmap(path)
    detect_symbols = min(args.detect_symbols, meta.preamble_len)

    period_samples = args.packet_period_samples
    if period_samples is None:
        period_samples = int(round(args.packet_period_seconds * runtime.sample_rate))
    if period_samples <= 0:
        raise ValueError("Packet period must be positive.")

    print(
        f"{path.name}: sf={meta.sf}, preamble={meta.preamble_len}, "
        f"samples={iq.num_complex}, symbol_samples={runtime.symbol_samples}, "
        f"period_samples={period_samples}"
    )

    seed_candidate, seed_row, checked = find_seed_packet(iq, runtime, meta, offsets, args)
    if seed_candidate is None or seed_row is None:
        print(f"  detected valid packets: 0 (seed scan checked {checked})")
        return []

    rows = [seed_row]
    max_rows = args.max_packets_per_file if args.max_packets_per_file else math.inf
    max_start = iq.num_complex - meta.preamble_len * runtime.symbol_samples
    period_index = 1
    attempts = 0
    misses = 0

    while len(rows) < max_rows:
        expected_start = seed_candidate.sample_start + period_index * period_samples
        if expected_start > max_start:
            break

        refined = refine_candidate_window(
            iq,
            int(expected_start),
            args.period_search_samples,
            runtime,
            detect_symbols,
            args,
        )
        attempts += 1
        if refined is not None and candidate_passes(
            refined,
            args.detect_threshold_db,
            args.peak_std_max,
            args.detection_mode != "noncoherent",
        ):
            candidate = rewind_to_first_preamble_symbol(
                iq,
                refined,
                runtime,
                meta.preamble_len,
                args.downsample,
                args.remove_dc,
                args.peak_width_db,
                args.detect_threshold_db,
                args.peak_bin_tolerance,
            )
            row = extract_valid_packet_features(
                iq,
                candidate,
                runtime,
                meta,
                len(rows),
                offsets,
                args,
            )
            if row is not None:
                rows.append(row)
                misses = 0
            else:
                misses += 1
        else:
            misses += 1

        if args.max_period_misses and misses >= args.max_period_misses:
            break
        period_index += 1

    print(
        f"  detected valid packets: {len(rows)} "
        f"(seed scan checked {checked}, periodic attempts {attempts})"
    )
    return rows


def collect_input_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(input_path.glob(pattern))


def write_rows(path: Path, rows: Iterable[dict], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_header(offsets: Sequence[int], phase_mode: str) -> List[str]:
    base = [
        "file_name",
        "experiment_id",
        "corridor_id",
        "position_id",
        "filename_sf",
        "filename_tx_power_dbm",
        "filename_preamble_len",
        "packet_index",
        "sample_start",
        "sample_rate",
        "bandwidth",
        "fft_size",
        "symbol_samples",
        "skip_preamble_symbols",
        "feature_symbols",
        "feature_bin_count",
        "normalize",
        "phase_mode",
        "preamble_avg_power_db",
        "preamble_peak_to_residual_db",
        "preamble_peak_width_3db_bins_avg",
        "preamble_peak_bin_mean",
        "preamble_peak_bin_std",
        "detect_score_db",
        "detect_peak_bin_mean",
        "detect_peak_bin_std",
    ]
    mag_cols = [f"preamble_fft_mag_bin_{offset_suffix(offset)}" for offset in offsets]
    phase_cols = [f"preamble_fft_phase_bin_{offset_suffix(offset)}" for offset in offsets]
    raw_phase_cols = []
    if phase_mode == "both":
        raw_phase_cols = [f"preamble_fft_phase_raw_bin_{offset_suffix(offset)}" for offset in offsets]
    return base + mag_cols + phase_cols + raw_phase_cols


def process_file(path: Path, args: argparse.Namespace, offsets: Sequence[int]) -> List[dict]:
    if args.periodic_extract:
        return process_file_periodic(path, args, offsets)

    meta = parse_file_meta(path, args.sf, args.preamble_len)
    runtime = build_runtime(meta.sf, args.sample_rate, args.bandwidth)
    iq = IQMemmap(path)

    print(
        f"{path.name}: sf={meta.sf}, preamble={meta.preamble_len}, "
        f"samples={iq.num_complex}, symbol_samples={runtime.symbol_samples}"
    )
    candidates = detect_packets(iq, runtime, meta.preamble_len, args)
    print(f"  detected packets: {len(candidates)}")

    rows = []
    for packet_index, candidate in enumerate(candidates):
        if args.plot_dir is not None and packet_index < args.plot_packets_per_file:
            plot_packet_diagnostics(iq, candidate, runtime, meta, packet_index, args)
        row = extract_valid_packet_features(
            iq,
            candidate,
            runtime,
            meta,
            len(rows),
            offsets,
            args,
        )
        if row is not None:
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract LoRa preamble dechirp FFT magnitude/phase features from "
            "USRP fc32 interleaved IQ .bin files."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw/usrp"))
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--output-csv", type=Path, default=Path("data/processedData/usrp_preamble_fft_features.csv"))
    parser.add_argument("--sample-rate", type=float, default=500_000.0)
    parser.add_argument("--bandwidth", type=float, default=125_000.0)
    parser.add_argument("--sf", type=int, default=None, help="Override SF instead of parsing it from file names.")
    parser.add_argument("--preamble-len", type=int, default=None, help="Override preamble length from file names.")
    parser.add_argument("--bin-count", type=int, default=16, help="Total FFT bins around aligned bin0, e.g. 8 or 16.")
    parser.add_argument("--feature-symbols", type=int, default=None, help="Preamble symbols used for feature averaging.")
    parser.add_argument("--skip-preamble-symbols", type=int, default=0, help="Skip unstable leading preamble symbols before feature averaging.")
    parser.add_argument("--detect-symbols", type=int, default=4, help="Consecutive upchirps used during detection.")
    parser.add_argument(
        "--detection-mode",
        choices=("per-symbol", "noncoherent"),
        default="per-symbol",
        help="Packet detector scoring mode. noncoherent sums dechirp FFT power across preamble symbols.",
    )
    parser.add_argument(
        "--feature-alignment",
        choices=("per-symbol", "noncoherent"),
        default="per-symbol",
        help="FFT-bin alignment used while extracting local mag/phase features.",
    )
    parser.add_argument("--scan-step-symbols", type=float, default=0.125, help="Coarse scan step measured in LoRa symbols.")
    parser.add_argument("--refine-step-samples", type=int, default=None, help="Fine sync step in samples. Default: oversampling factor.")
    parser.add_argument("--suppress-symbols", type=float, default=12.0, help="Minimum spacing between packet detections.")
    parser.add_argument("--detect-threshold-db", type=float, default=-6.0)
    parser.add_argument("--peak-std-max", type=float, default=3.0)
    parser.add_argument("--peak-bin-tolerance", type=float, default=4.0)
    parser.add_argument("--peak-width-db", type=float, default=-3.0)
    parser.add_argument("--normalize", choices=("none", "peak", "energy"), default="peak")
    parser.add_argument("--phase-mode", choices=("relative", "raw", "both"), default="relative")
    parser.add_argument("--downsample", choices=("mean", "pick"), default="mean")
    parser.add_argument("--remove-dc", action="store_true")
    parser.add_argument("--power-symbols", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-packets-per-file", type=int, default=None)
    parser.add_argument("--max-scan-symbols", type=int, default=None, help="Debug limit for scanning only the first N symbols.")
    parser.add_argument(
        "--no-verify-peak-std",
        dest="verify_peak_std",
        action="store_false",
        help="Do not reject extracted rows based on feature-window per-symbol peak-bin spread.",
    )
    parser.set_defaults(verify_peak_std=True)
    parser.add_argument("--verify-peak-std-max", type=float, default=2.0, help="Reject extracted packets whose full feature-window peak bins are not stable.")
    parser.add_argument("--verify-score-db", type=float, default=0.0, help="Reject extracted packets below this full feature-window peak-to-residual score.")
    parser.add_argument("--periodic-extract", action="store_true", help="Use fixed packet period after finding a valid seed packet.")
    parser.add_argument("--packet-period-samples", type=int, default=None)
    parser.add_argument("--packet-period-seconds", type=float, default=5.0)
    parser.add_argument("--period-search-samples", type=int, default=50_000)
    parser.add_argument("--max-period-misses", type=int, default=4)
    parser.add_argument("--seed-max-scan-symbols", type=int, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None, help="Write diagnostic STFT/dechirp FFT PNGs here.")
    parser.add_argument("--plot-style", choices=("matplotlib", "paper", "debug"), default="matplotlib")
    parser.add_argument("--plot-packets-per-file", type=int, default=3)
    parser.add_argument("--plot-symbols", type=int, default=None, help="Preamble symbols shown in diagnostic plots.")
    parser.add_argument("--plot-start-symbol", type=int, default=None, help="First preamble symbol shown in plots. Default: skip value for paper style, 0 for debug style.")
    parser.add_argument("--plot-event-symbols", type=float, default=None, help="Symbols shown in the located STFT event plot.")
    parser.add_argument("--plot-bin-count", type=int, default=33, help="Local aligned FFT bins shown in diagnostic plots.")
    parser.add_argument("--plot-profile-bins", type=int, default=100, help="Half-width of signed DFT bins shown in line-profile plots.")
    parser.add_argument("--stft-nfft", type=int, default=1024)
    parser.add_argument("--stft-hop", type=int, default=128)
    parser.add_argument("--stft-floor-db", type=float, default=-75.0)
    parser.add_argument("--stft-ref-percentile", type=float, default=99.7)
    parser.add_argument("--profile-floor-db", type=float, default=-55.0)
    parser.add_argument("--matplotlib-cmap", default="viridis")
    parser.add_argument("--paper-cmap", default="turbo")
    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = collect_input_files(args.input, args.glob)
    if args.max_files is not None:
        input_files = input_files[: args.max_files]
    if not input_files:
        raise FileNotFoundError(f"No files matched {args.input} / {args.glob}")

    offsets = make_bin_offsets(args.bin_count)
    header = build_header(offsets, args.phase_mode)
    all_rows: List[dict] = []

    for path in input_files:
        all_rows.extend(process_file(path, args, offsets))

    write_rows(args.output_csv, all_rows, header)
    print(f"Wrote {len(all_rows)} packet rows to {args.output_csv}")


if __name__ == "__main__":
    main()
