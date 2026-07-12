from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

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
    symbol_scores_db: list[float]
    symbol_peak_bins: list[int]


class IQMemmap:
    """Read interleaved little-endian float32 I/Q without loading a full file."""

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
            "This script expects integer oversampling: "
            f"sample_rate / bandwidth = {sample_rate} / {bandwidth} = {os_float:.6f}."
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


def read_dechirped_symbol(
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
    return baseband * runtime.downchirp


def dechirp_fft(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    downsample_method: str,
    remove_dc: bool,
) -> Optional[np.ndarray]:
    dechirped = read_dechirped_symbol(iq, start, runtime, downsample_method, remove_dc)
    if dechirped is None:
        return None
    return np.fft.fft(dechirped, n=runtime.fft_size)


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
    return power_spectrum_metrics(np.abs(spec) ** 2, peak_width_db)


def evaluate_candidate(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    symbol_count: int,
    downsample_method: str,
    remove_dc: bool,
    peak_width_db: float,
) -> Optional[Candidate]:
    scores: list[float] = []
    peaks: list[int] = []

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
        peak_bin, score_db, _ = spectrum_metrics(spec, peak_width_db)
        scores.append(score_db)
        peaks.append(peak_bin)

    return Candidate(
        sample_start=start,
        score_db=float(np.mean(scores)),
        peak_bin_mean=circular_mean_bin(peaks, runtime.fft_size),
        peak_bin_std=circular_std_bins(peaks, runtime.fft_size),
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
    powers: list[np.ndarray] = []
    peaks: list[int] = []
    scores: list[float] = []

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
        peak_bin, score_db, _ = power_spectrum_metrics(power, peak_width_db)
        powers.append(power)
        peaks.append(peak_bin)
        scores.append(score_db)

    summed_power = np.sum(np.vstack(powers), axis=0)
    common_peak, common_score_db, _ = power_spectrum_metrics(summed_power, peak_width_db)
    return Candidate(
        sample_start=start,
        score_db=float(common_score_db),
        peak_bin_mean=float(common_peak),
        peak_bin_std=circular_std_bins(peaks, runtime.fft_size),
        symbol_scores_db=scores,
        symbol_peak_bins=peaks,
    )


def evaluate_candidate_for_detection(
    iq: IQMemmap,
    start: int,
    runtime: LoraRuntime,
    symbol_count: int,
    args: argparse.Namespace,
) -> Optional[Candidate]:
    if args.detection_mode == "noncoherent":
        return evaluate_candidate_noncoherent(
            iq,
            start,
            runtime,
            symbol_count,
            args.downsample,
            args.remove_dc,
            args.peak_width_db,
        )
    return evaluate_candidate(
        iq,
        start,
        runtime,
        symbol_count,
        args.downsample,
        args.remove_dc,
        args.peak_width_db,
    )


def candidate_passes(candidate: Candidate, args: argparse.Namespace) -> bool:
    if candidate.score_db < args.detect_threshold_db:
        return False
    if args.detection_mode != "noncoherent" and candidate.peak_bin_std > args.peak_std_max:
        return False
    return True


def refine_candidate(
    iq: IQMemmap,
    coarse_start: int,
    scan_step: int,
    runtime: LoraRuntime,
    detect_symbols: int,
    args: argparse.Namespace,
) -> Optional[Candidate]:
    refine_step = max(1, args.refine_step_samples or runtime.os_factor)
    max_start = iq.num_complex - detect_symbols * runtime.symbol_samples
    low = max(0, coarse_start - scan_step)
    high = min(max_start, coarse_start + scan_step)
    best: Optional[Candidate] = None

    for start in range(low, high + 1, refine_step):
        candidate = evaluate_candidate_for_detection(iq, start, runtime, detect_symbols, args)
        if candidate is None:
            continue
        if best is None or candidate.score_db > best.score_db:
            best = candidate
    return best


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
        candidate = evaluate_candidate_for_detection(iq, start, runtime, detect_symbols, args)
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
    args: argparse.Namespace,
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
            args.downsample,
            args.remove_dc,
            args.peak_width_db,
        )
        if previous_candidate is None or previous_candidate.score_db < args.detect_threshold_db:
            break
        if circular_distance_bins(previous_candidate.peak_bin_mean, reference_peak, runtime.fft_size) > args.peak_bin_tolerance:
            break
        start = previous

    rewound = evaluate_candidate(
        iq,
        start,
        runtime,
        len(candidate.symbol_scores_db),
        args.downsample,
        args.remove_dc,
        args.peak_width_db,
    )
    return rewound if rewound is not None else candidate


def detect_packets(iq: IQMemmap, runtime: LoraRuntime, preamble_len: int, args: argparse.Namespace) -> list[Candidate]:
    scan_step = max(1, int(round(args.scan_step_symbols * runtime.symbol_samples)))
    suppress_samples = max(runtime.symbol_samples, int(round(args.suppress_symbols * runtime.symbol_samples)))
    detect_symbols = min(args.detect_symbols, preamble_len)
    max_scan_start = iq.num_complex - detect_symbols * runtime.symbol_samples
    if args.max_scan_symbols is not None:
        max_scan_start = min(max_scan_start, args.max_scan_symbols * runtime.symbol_samples)

    packets: list[Candidate] = []
    start = 0
    checked = 0

    while start <= max_scan_start:
        candidate = evaluate_candidate_for_detection(iq, start, runtime, detect_symbols, args)
        checked += 1

        if candidate is not None and candidate_passes(candidate, args):
            refined = refine_candidate(iq, start, scan_step, runtime, detect_symbols, args)
            if refined is not None:
                candidate = rewind_to_first_preamble_symbol(iq, refined, runtime, preamble_len, args)

            if candidate_passes(candidate, args):
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


def detect_packets_periodic(
    iq: IQMemmap,
    runtime: LoraRuntime,
    preamble_len: int,
    args: argparse.Namespace,
) -> list[Candidate]:
    seed_args = copy.copy(args)
    seed_args.max_packets_per_file = 1
    if args.seed_max_scan_symbols is not None:
        seed_args.max_scan_symbols = args.seed_max_scan_symbols

    seed_candidates = detect_packets(iq, runtime, preamble_len, seed_args)
    if not seed_candidates:
        return []

    period_samples = args.packet_period_samples
    if period_samples is None:
        period_samples = int(round(args.packet_period_seconds * runtime.sample_rate))
    if period_samples <= 0:
        raise ValueError("Packet period must be positive.")

    detect_symbols = min(args.detect_symbols, preamble_len)
    max_start = iq.num_complex - preamble_len * runtime.symbol_samples
    packets = [seed_candidates[0]]
    period_index = 1
    misses = 0
    max_packets = args.max_packets_per_file or math.inf

    while len(packets) < max_packets:
        expected_start = packets[0].sample_start + period_index * period_samples
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
        if refined is not None and candidate_passes(refined, args):
            candidate = rewind_to_first_preamble_symbol(iq, refined, runtime, preamble_len, args)
            if candidate_passes(candidate, args):
                packets.append(candidate)
                misses = 0
            else:
                misses += 1
        else:
            misses += 1
        if args.max_period_misses and misses >= args.max_period_misses:
            break
        period_index += 1

    return packets


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_q_values(text: str) -> list[int]:
    values = parse_int_list(text)
    for q in values:
        if q <= 0:
            raise ValueError("All q values must be positive.")
    return values


def collect_input_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(input_path.glob(pattern))


def load_packet_starts_csv(path: Path) -> dict[str, list[dict]]:
    starts_by_file: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"file_name", "sample_start"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader):
            try:
                sample_start = int(float(row["sample_start"]))
            except (TypeError, ValueError):
                continue
            packet_index_text = row.get("packet_index", row_number)
            try:
                packet_index = int(float(packet_index_text))
            except (TypeError, ValueError):
                packet_index = row_number
            starts_by_file[Path(row["file_name"]).name].append(
                {
                    "sample_start": sample_start,
                    "packet_index": packet_index,
                    "source_row": row_number,
                }
            )
    for rows in starts_by_file.values():
        rows.sort(key=lambda row: (row["packet_index"], row["sample_start"]))
    return dict(starts_by_file)


def offset_suffix_float(value: float) -> str:
    text = f"{value:+.6f}".rstrip("0").rstrip(".")
    return text


def parabolic_delta(log_mag: np.ndarray, peak_idx: int) -> tuple[float, bool]:
    if peak_idx <= 0 or peak_idx >= log_mag.size - 1:
        return 0.0, False
    left = float(log_mag[peak_idx - 1])
    center = float(log_mag[peak_idx])
    right = float(log_mag[peak_idx + 1])
    denom = left - 2.0 * center + right
    if abs(denom) < EPS:
        return 0.0, False
    delta = 0.5 * (left - right) / denom
    if not math.isfinite(delta):
        return 0.0, False
    return float(delta), True


def local_width_3db_bins(db_rel_peak: np.ndarray, peak_idx: int, q: int) -> float:
    width = 1
    for idx in range(peak_idx - 1, -1, -1):
        if db_rel_peak[idx] < -3.0:
            break
        width += 1
    for idx in range(peak_idx + 1, db_rel_peak.size):
        if db_rel_peak[idx] < -3.0:
            break
        width += 1
    return float(width / q)


def normalize_complex(values: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    if mode == "none":
        return values.astype(np.complex128), 1.0
    if mode == "energy":
        ref = math.sqrt(float(np.sum(np.abs(values) ** 2)) + EPS)
        return values / ref, ref
    if mode == "center":
        ref = float(abs(values[values.size // 2])) + EPS
        return values / ref, ref
    ref = float(np.max(np.abs(values))) + EPS
    return values / ref, ref


def analyze_zero_padded_symbol(
    dechirped: np.ndarray,
    runtime: LoraRuntime,
    q: int,
    align_bin: int,
    symbol_info: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict, tuple[np.ndarray, np.ndarray]]:
    zp_size = q * runtime.fft_size
    center_idx = (align_bin * q) % zp_size
    half_width_zp = int(round(args.window_original_bins * q))
    offsets_zp = np.arange(-half_width_zp, half_width_zp + 1, dtype=np.int64)
    subbin_offsets = offsets_zp.astype(np.float64) / float(q)
    indices = np.mod(center_idx + offsets_zp, zp_size)

    spectrum = np.fft.fft(dechirped, n=zp_size)
    local_complex = spectrum[indices].astype(np.complex128)
    center_complex = spectrum[center_idx]
    phase_centered = local_complex * np.exp(-1j * np.angle(center_complex))
    norm_complex, norm_ref = normalize_complex(phase_centered, args.normalize)

    mag_raw = np.abs(local_complex)
    local_peak_idx = int(np.argmax(mag_raw))
    local_peak_mag = float(mag_raw[local_peak_idx]) + EPS
    mag_db_rel_peak = 20.0 * np.log10(np.maximum(mag_raw, EPS) / local_peak_mag)
    mag_norm = np.abs(norm_complex)
    power_raw = mag_raw ** 2
    total_power = float(np.sum(power_raw)) + EPS

    log_mag = 20.0 * np.log10(np.maximum(mag_raw, EPS))
    interp_delta_zp, interp_ok = parabolic_delta(log_mag, local_peak_idx)
    peak_offset_zp = int(offsets_zp[local_peak_idx])
    peak_offset_bins = peak_offset_zp / float(q)
    interpolated_peak_offset_bins = peak_offset_bins + interp_delta_zp / float(q)

    left_power = float(np.sum(power_raw[subbin_offsets < 0.0]))
    right_power = float(np.sum(power_raw[subbin_offsets > 0.0]))
    side_mask = np.abs(subbin_offsets) > args.side_exclusion_bins
    side_power_fraction = float(np.sum(power_raw[side_mask]) / total_power)
    asymmetry = float((right_power - left_power) / (right_power + left_power + EPS))

    exclude_secondary = np.abs(subbin_offsets - interpolated_peak_offset_bins) <= args.secondary_exclusion_bins
    secondary_mask = ~exclude_secondary
    if np.any(secondary_mask):
        secondary_candidates = np.where(secondary_mask)[0]
        secondary_idx = int(secondary_candidates[np.argmax(mag_raw[secondary_candidates])])
        secondary_offset = float(subbin_offsets[secondary_idx])
        secondary_rel_db = float(mag_db_rel_peak[secondary_idx])
    else:
        secondary_offset = float("nan")
        secondary_rel_db = float("nan")

    curvature_db = float("nan")
    if 0 < local_peak_idx < log_mag.size - 1:
        curvature_db = float(log_mag[local_peak_idx - 1] - 2.0 * log_mag[local_peak_idx] + log_mag[local_peak_idx + 1])

    rows: list[dict] = []
    for idx, offset_zp in enumerate(offsets_zp):
        row = {
            **symbol_info,
            "q": q,
            "zp_fft_size": zp_size,
            "align_bin": align_bin,
            "k_center": int(center_idx),
            "k_offset_zp": int(offset_zp),
            "subbin_offset": float(subbin_offsets[idx]),
            "subbin_offset_label": offset_suffix_float(float(subbin_offsets[idx])),
            "mag_raw": float(mag_raw[idx]),
            "mag_norm": float(mag_norm[idx]),
            "mag_db_rel_peak": float(mag_db_rel_peak[idx]),
            "phase_rad_rel_center": float(np.angle(phase_centered[idx])),
            "real_norm": float(np.real(norm_complex[idx])),
            "imag_norm": float(np.imag(norm_complex[idx])),
        }
        rows.append(row)

    summary = {
        **symbol_info,
        "q": q,
        "zp_fft_size": zp_size,
        "align_bin": align_bin,
        "k_center": int(center_idx),
        "norm_ref": float(norm_ref),
        "peak_offset_zp": peak_offset_zp,
        "peak_offset_bins": float(peak_offset_bins),
        "parabolic_delta_zp": float(interp_delta_zp),
        "interpolated_peak_offset_bins": float(interpolated_peak_offset_bins),
        "parabolic_ok": int(interp_ok),
        "local_peak_mag_raw": float(local_peak_mag),
        "local_peak_width_3db_bins": local_width_3db_bins(mag_db_rel_peak, local_peak_idx, q),
        "side_power_fraction": side_power_fraction,
        "left_power_fraction": float(left_power / total_power),
        "right_power_fraction": float(right_power / total_power),
        "asymmetry": asymmetry,
        "curvature_db_per_zp_bin": curvature_db,
        "secondary_peak_offset_bins": secondary_offset,
        "secondary_peak_rel_db": secondary_rel_db,
    }
    return rows, summary, (subbin_offsets, mag_db_rel_peak)


def choose_align_bins(dechirped_symbols: Sequence[np.ndarray], runtime: LoraRuntime, args: argparse.Namespace) -> tuple[list[int], list[int], list[float]]:
    original_peak_bins: list[int] = []
    original_scores: list[float] = []
    powers: list[np.ndarray] = []

    for dechirped in dechirped_symbols:
        spec = np.fft.fft(dechirped, n=runtime.fft_size)
        power = np.abs(spec) ** 2
        peak_bin, score_db, _ = power_spectrum_metrics(power, args.peak_width_db)
        original_peak_bins.append(peak_bin)
        original_scores.append(score_db)
        powers.append(power)

    if args.profile_alignment == "bin0":
        align_bins = [0] * len(dechirped_symbols)
    elif args.profile_alignment == "noncoherent":
        common_peak = int(np.argmax(np.sum(np.vstack(powers), axis=0)))
        align_bins = [common_peak] * len(dechirped_symbols)
    else:
        align_bins = original_peak_bins
    return align_bins, original_peak_bins, original_scores


def aggregate_packet_summaries(symbol_summaries: Sequence[dict], meta_fields: dict, q_values: Sequence[int]) -> list[dict]:
    numeric_fields = [
        "score_db",
        "original_peak_bin",
        "align_bin",
        "peak_offset_bins",
        "interpolated_peak_offset_bins",
        "local_peak_width_3db_bins",
        "side_power_fraction",
        "left_power_fraction",
        "right_power_fraction",
        "asymmetry",
        "curvature_db_per_zp_bin",
        "secondary_peak_offset_bins",
        "secondary_peak_rel_db",
    ]
    rows: list[dict] = []
    for q in q_values:
        q_rows = [row for row in symbol_summaries if row["q"] == q]
        if not q_rows:
            continue
        out = {
            **meta_fields,
            "q": q,
            "zp_fft_size": q_rows[0]["zp_fft_size"],
            "symbol_count": len(q_rows),
        }
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in q_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                out[f"{field}_mean"] = float(np.mean(values))
                out[f"{field}_std"] = float(np.std(values))
            else:
                out[f"{field}_mean"] = float("nan")
                out[f"{field}_std"] = float("nan")
        rows.append(out)
    return rows


def extract_packet_profiles(
    iq: IQMemmap,
    candidate: Candidate,
    runtime: LoraRuntime,
    meta: FileMeta,
    packet_index: int,
    q_values: Sequence[int],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict], dict[int, list[tuple[np.ndarray, np.ndarray]]]]:
    skip_symbols = max(0, min(args.skip_preamble_symbols, meta.preamble_len - 1))
    available_symbols = meta.preamble_len - skip_symbols
    symbol_count = args.feature_symbols if args.feature_symbols is not None else available_symbols
    symbol_count = max(1, min(symbol_count, available_symbols))

    dechirped_symbols: list[np.ndarray] = []
    preamble_symbol_indices: list[int] = []
    for preamble_symbol_index in range(skip_symbols, skip_symbols + symbol_count):
        start = candidate.sample_start + preamble_symbol_index * runtime.symbol_samples
        dechirped = read_dechirped_symbol(iq, start, runtime, args.downsample, args.remove_dc)
        if dechirped is None:
            break
        dechirped_symbols.append(dechirped)
        preamble_symbol_indices.append(preamble_symbol_index)

    if not dechirped_symbols:
        raise RuntimeError(f"Could not extract preamble symbols at sample {candidate.sample_start}")

    align_bins, original_peak_bins, original_scores = choose_align_bins(dechirped_symbols, runtime, args)
    packet_fields = {
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
        "feature_symbols": len(dechirped_symbols),
        "profile_alignment": args.profile_alignment,
        "normalize": args.normalize,
        "detect_score_db": candidate.score_db,
        "detect_peak_bin_mean": candidate.peak_bin_mean,
        "detect_peak_bin_std": candidate.peak_bin_std,
    }

    long_rows: list[dict] = []
    symbol_summaries: list[dict] = []
    plot_profiles: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {q: [] for q in q_values}

    for local_symbol_idx, (dechirped, preamble_symbol_index, align_bin, original_peak_bin, score_db) in enumerate(
        zip(dechirped_symbols, preamble_symbol_indices, align_bins, original_peak_bins, original_scores)
    ):
        symbol_info = {
            **packet_fields,
            "local_symbol_index": local_symbol_idx,
            "preamble_symbol_index": preamble_symbol_index,
            "symbol_sample_start": candidate.sample_start + preamble_symbol_index * runtime.symbol_samples,
            "original_peak_bin": original_peak_bin,
            "score_db": float(score_db),
        }
        for q in q_values:
            rows, summary, profile = analyze_zero_padded_symbol(dechirped, runtime, q, align_bin, symbol_info, args)
            long_rows.extend(rows)
            symbol_summaries.append(summary)
            plot_profiles[q].append(profile)

    packet_summaries = aggregate_packet_summaries(symbol_summaries, packet_fields, q_values)
    return long_rows, symbol_summaries, packet_summaries, plot_profiles


def plot_packet_profiles(
    output_dir: Path,
    meta: FileMeta,
    packet_index: int,
    sample_start: int,
    plot_profiles: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    args: argparse.Namespace,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  matplotlib unavailable, writing SVG plot instead ({exc})")
        return write_svg_packet_profiles(output_dir, meta, packet_index, sample_start, plot_profiles, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for q in sorted(plot_profiles):
        profiles = plot_profiles[q]
        if not profiles:
            continue
        x = profiles[0][0]
        y = np.vstack([profile[1] for profile in profiles])
        y_mean = np.maximum(np.mean(y, axis=0), args.plot_floor_db)
        ax.plot(x, y_mean, label=f"q={q}", linewidth=1.4)

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_xlim(-args.window_original_bins, args.window_original_bins)
    ax.set_ylim(args.plot_floor_db, 2.0)
    ax.set_xlabel("sub-bin offset from aligned LoRa bin")
    ax.set_ylabel("magnitude relative to local peak (dB)")
    ax.set_title(f"{meta.file_name} packet {packet_index} zero-padding FFT profile")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", ncol=min(3, max(1, len(plot_profiles))))
    fig.tight_layout()
    path = output_dir / f"{Path(meta.file_name).stem}_pkt{packet_index:03d}_s{sample_start}_zp_fft_profile.png"
    fig.savefig(path, dpi=args.plot_dpi)
    plt.close(fig)
    return path


def write_svg_packet_profiles(
    output_dir: Path,
    meta: FileMeta,
    packet_index: int,
    sample_start: int,
    plot_profiles: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    args: argparse.Namespace,
) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    width = 1000
    height = 540
    left = 78
    right = 28
    top = 48
    bottom = 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = -float(args.window_original_bins)
    x_max = float(args.window_original_bins)
    y_min = float(args.plot_floor_db)
    y_max = 2.0
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        value = min(max(value, y_min), y_max)
        return top + (y_max - value) / (y_max - y_min) * plot_h

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-family="Arial" font-size="18" fill="#222">'
        f'{html.escape(meta.file_name)} packet {packet_index} zero-padding FFT profile</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#bbb"/>',
    ]

    for tick in range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1):
        x = sx(float(tick))
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e6e6e6"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{height - 42}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#444">{tick}</text>'
        )

    y_step = 10.0
    tick = math.ceil(y_min / y_step) * y_step
    while tick <= y_max:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#444">{tick:g}</text>'
        )
        tick += y_step

    zero_x = sx(0.0)
    lines.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{top + plot_h}" stroke="#333" stroke-dasharray="5 5"/>')

    legend_y = top + 18
    for idx, q in enumerate(sorted(plot_profiles)):
        profiles = plot_profiles[q]
        if not profiles:
            continue
        x_values = profiles[0][0]
        y_values = np.maximum(np.mean(np.vstack([profile[1] for profile in profiles]), axis=0), args.plot_floor_db)
        points = " ".join(f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(x_values, y_values))
        color = colors[idx % len(colors)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = left + 18 + idx * 92
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial" '
            f'font-size="13" fill="#222">q={q}</text>'
        )

    lines.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 15}" text-anchor="middle" '
        f'font-family="Arial" font-size="13" fill="#222">sub-bin offset from aligned LoRa bin</text>'
    )
    lines.append(
        f'<text x="20" y="{top + plot_h / 2:.2f}" transform="rotate(-90 20 {top + plot_h / 2:.2f})" '
        f'text-anchor="middle" font-family="Arial" font-size="13" fill="#222">magnitude relative to local peak (dB)</text>'
    )
    lines.append("</svg>")

    path = output_dir / f"{Path(meta.file_name).stem}_pkt{packet_index:03d}_s{sample_start}_zp_fft_profile.svg"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_sample_start_candidates(
    iq: IQMemmap,
    runtime: LoraRuntime,
    preamble_len: int,
    starts: Sequence[int],
    args: argparse.Namespace,
) -> list[Candidate]:
    detect_symbols = min(args.detect_symbols, preamble_len)
    candidates: list[Candidate] = []
    for start in starts:
        candidate = evaluate_candidate_for_detection(iq, start, runtime, detect_symbols, args)
        if candidate is None:
            print(f"  skip explicit sample_start={start}: out of range")
            continue
        candidates.append(candidate)
    return candidates


def open_writer(path: Path, fieldnames: Sequence[str]) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    return writer, handle


def maybe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def write_point_summary(packet_csv: Path, point_csv: Path) -> int:
    group_fields = [
        "experiment_id",
        "corridor_id",
        "position_id",
        "filename_sf",
        "filename_tx_power_dbm",
        "filename_preamble_len",
        "sample_rate",
        "bandwidth",
        "fft_size",
        "symbol_samples",
        "profile_alignment",
        "normalize",
        "q",
        "zp_fft_size",
    ]
    skip_numeric = {"packet_index", "sample_start"}

    with packet_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        packet_rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not packet_rows:
        point_csv.parent.mkdir(parents=True, exist_ok=True)
        with point_csv.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=group_fields + ["packet_count", "file_count", "file_names"]).writeheader()
        return 0

    numeric_fields: list[str] = []
    for field in fieldnames:
        if field in group_fields or field in skip_numeric:
            continue
        if any(maybe_float(row.get(field)) is not None for row in packet_rows):
            numeric_fields.append(field)

    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in packet_rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups[key].append(row)

    output_fields = group_fields + ["packet_count", "file_count", "file_names"]
    for field in numeric_fields:
        output_fields.extend(
            [
                f"{field}_point_mean",
                f"{field}_point_std",
                f"{field}_point_median",
                f"{field}_point_min",
                f"{field}_point_max",
            ]
        )

    point_csv.parent.mkdir(parents=True, exist_ok=True)
    with point_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            out = {field: value for field, value in zip(group_fields, key)}
            file_names = sorted({row.get("file_name", "") for row in rows if row.get("file_name", "")})
            out["packet_count"] = len(rows)
            out["file_count"] = len(file_names)
            out["file_names"] = ";".join(file_names)
            for field in numeric_fields:
                values = np.asarray(
                    [value for value in (maybe_float(row.get(field)) for row in rows) if value is not None],
                    dtype=np.float64,
                )
                if values.size:
                    out[f"{field}_point_mean"] = float(np.mean(values))
                    out[f"{field}_point_std"] = float(np.std(values))
                    out[f"{field}_point_median"] = float(np.median(values))
                    out[f"{field}_point_min"] = float(np.min(values))
                    out[f"{field}_point_max"] = float(np.max(values))
                else:
                    out[f"{field}_point_mean"] = float("nan")
                    out[f"{field}_point_std"] = float("nan")
                    out[f"{field}_point_median"] = float("nan")
                    out[f"{field}_point_min"] = float("nan")
                    out[f"{field}_point_max"] = float("nan")
            writer.writerow(out)
    return len(groups)


LONG_HEADER = [
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
    "profile_alignment",
    "normalize",
    "detect_score_db",
    "detect_peak_bin_mean",
    "detect_peak_bin_std",
    "local_symbol_index",
    "preamble_symbol_index",
    "symbol_sample_start",
    "original_peak_bin",
    "score_db",
    "q",
    "zp_fft_size",
    "align_bin",
    "k_center",
    "k_offset_zp",
    "subbin_offset",
    "subbin_offset_label",
    "mag_raw",
    "mag_norm",
    "mag_db_rel_peak",
    "phase_rad_rel_center",
    "real_norm",
    "imag_norm",
]

SYMBOL_SUMMARY_HEADER = [
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
    "profile_alignment",
    "normalize",
    "detect_score_db",
    "detect_peak_bin_mean",
    "detect_peak_bin_std",
    "local_symbol_index",
    "preamble_symbol_index",
    "symbol_sample_start",
    "original_peak_bin",
    "score_db",
    "q",
    "zp_fft_size",
    "align_bin",
    "k_center",
    "norm_ref",
    "peak_offset_zp",
    "peak_offset_bins",
    "parabolic_delta_zp",
    "interpolated_peak_offset_bins",
    "parabolic_ok",
    "local_peak_mag_raw",
    "local_peak_width_3db_bins",
    "side_power_fraction",
    "left_power_fraction",
    "right_power_fraction",
    "asymmetry",
    "curvature_db_per_zp_bin",
    "secondary_peak_offset_bins",
    "secondary_peak_rel_db",
]

PACKET_SUMMARY_BASE = [
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
    "profile_alignment",
    "normalize",
    "detect_score_db",
    "detect_peak_bin_mean",
    "detect_peak_bin_std",
    "q",
    "zp_fft_size",
    "symbol_count",
]

PACKET_SUMMARY_METRICS = [
    "score_db",
    "original_peak_bin",
    "align_bin",
    "peak_offset_bins",
    "interpolated_peak_offset_bins",
    "local_peak_width_3db_bins",
    "side_power_fraction",
    "left_power_fraction",
    "right_power_fraction",
    "asymmetry",
    "curvature_db_per_zp_bin",
    "secondary_peak_offset_bins",
    "secondary_peak_rel_db",
]

PACKET_SUMMARY_HEADER = PACKET_SUMMARY_BASE + [
    f"{metric}_{stat}" for metric in PACKET_SUMMARY_METRICS for stat in ("mean", "std")
]


def process_file(
    path: Path,
    q_values: Sequence[int],
    writers: tuple[csv.DictWriter, csv.DictWriter, csv.DictWriter],
    args: argparse.Namespace,
    packet_start_rows: Optional[Sequence[dict]] = None,
) -> tuple[int, int, int]:
    meta = parse_file_meta(path, args.sf, args.preamble_len)
    runtime = build_runtime(meta.sf, args.sample_rate, args.bandwidth)
    iq = IQMemmap(path)

    print(
        f"{path.name}: sf={meta.sf}, preamble={meta.preamble_len}, "
        f"samples={iq.num_complex}, symbol_samples={runtime.symbol_samples}"
    )

    explicit_packet_indices: Optional[list[int]] = None
    if packet_start_rows is not None:
        candidates = make_sample_start_candidates(
            iq,
            runtime,
            meta.preamble_len,
            [int(row["sample_start"]) for row in packet_start_rows],
            args,
        )
        explicit_packet_indices = [int(row["packet_index"]) for row in packet_start_rows]
    elif args.sample_starts:
        candidates = make_sample_start_candidates(
            iq,
            runtime,
            meta.preamble_len,
            parse_int_list(args.sample_starts),
            args,
        )
    elif args.periodic_extract:
        candidates = detect_packets_periodic(iq, runtime, meta.preamble_len, args)
    else:
        candidates = detect_packets(iq, runtime, meta.preamble_len, args)

    if args.max_packets_per_file is not None and args.max_packets_per_file > 0:
        candidates = candidates[: args.max_packets_per_file]
    print(f"  selected packets: {len(candidates)}")

    long_writer, symbol_writer, packet_writer = writers
    long_count = 0
    symbol_count = 0
    packet_count = 0

    for candidate_idx, candidate in enumerate(candidates):
        packet_index = (
            explicit_packet_indices[candidate_idx]
            if explicit_packet_indices is not None and candidate_idx < len(explicit_packet_indices)
            else candidate_idx
        )
        try:
            long_rows, symbol_rows, packet_rows, plot_profiles = extract_packet_profiles(
                iq,
                candidate,
                runtime,
                meta,
                packet_index,
                q_values,
                args,
            )
        except RuntimeError as exc:
            print(f"  skip packet {packet_index}: {exc}")
            continue

        if args.min_packet_score_db is not None:
            score_values = [
                maybe_float(row.get("score_db_mean"))
                for row in packet_rows
                if maybe_float(row.get("score_db_mean")) is not None
            ]
            if not score_values or max(score_values) < args.min_packet_score_db:
                if args.verbose:
                    score_text = "nan" if not score_values else f"{max(score_values):.3f}"
                    print(
                        f"  reject packet {packet_index} at {candidate.sample_start}: "
                        f"score_db_mean {score_text} < {args.min_packet_score_db:.3f}"
                    )
                continue

        long_writer.writerows(long_rows)
        symbol_writer.writerows(symbol_rows)
        packet_writer.writerows(packet_rows)
        long_count += len(long_rows)
        symbol_count += len(symbol_rows)
        packet_count += len(packet_rows)

        if args.plot_packets_per_file and packet_index < args.plot_packets_per_file:
            plot_path = plot_packet_profiles(
                args.output_dir / "plots",
                meta,
                packet_index,
                candidate.sample_start,
                plot_profiles,
                args,
            )
            if plot_path is not None:
                print(f"  plot: {plot_path}")

    return long_count, symbol_count, packet_count


def write_run_config(path: Path, args: argparse.Namespace, q_values: Sequence[int], input_files: Sequence[Path]) -> None:
    config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "q_values": list(q_values),
        "input_files": [str(path) for path in input_files],
        "notes": [
            "Zero-padding interpolates the dechirped symbol spectrum; it does not create true resolution beyond the LoRa bandwidth.",
            "Use interpolated_peak_offset_bins/asymmetry/secondary_peak_rel_db as shape diagnostics, then compare across locations or packets.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-padded FFT on LoRa preamble I/Q symbols after dechirping, "
            "then export sub-bin spectra and peak-shape diagnostics."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("INFOCOM_origin_data/LoRaRSSI+IQ_data"))
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--output-dir", type=Path, default=Path("v2_output/20260624_zero_padding_fft"))
    parser.add_argument(
        "--packet-starts-csv",
        type=Path,
        default=None,
        help="Optional CSV with file_name and sample_start columns from a previous trusted extractor.",
    )
    parser.add_argument("--sample-rate", type=float, default=500_000.0)
    parser.add_argument("--bandwidth", type=float, default=125_000.0)
    parser.add_argument("--sf", type=int, default=None, help="Override SF instead of parsing it from file names.")
    parser.add_argument("--preamble-len", type=int, default=None, help="Override preamble length from file names.")
    parser.add_argument("--q-values", default="1,4,8,16,32", help="Comma-separated zero-padding factors.")
    parser.add_argument("--window-original-bins", type=float, default=2.0, help="Keep +/- this many original LoRa bins.")
    parser.add_argument("--feature-symbols", type=int, default=8, help="Preamble symbols used for profile averaging.")
    parser.add_argument("--skip-preamble-symbols", type=int, default=1, help="Skip unstable leading preamble symbols.")
    parser.add_argument("--profile-alignment", choices=("per-symbol", "noncoherent", "bin0"), default="per-symbol")
    parser.add_argument("--normalize", choices=("peak", "center", "energy", "none"), default="peak")
    parser.add_argument("--downsample", choices=("mean", "pick"), default="mean")
    parser.add_argument("--remove-dc", action="store_true")

    parser.add_argument("--detect-symbols", type=int, default=4)
    parser.add_argument("--detection-mode", choices=("per-symbol", "noncoherent"), default="per-symbol")
    parser.add_argument("--scan-step-symbols", type=float, default=0.125)
    parser.add_argument("--refine-step-samples", type=int, default=None)
    parser.add_argument("--suppress-symbols", type=float, default=12.0)
    parser.add_argument("--detect-threshold-db", type=float, default=-6.0)
    parser.add_argument("--peak-std-max", type=float, default=3.0)
    parser.add_argument("--peak-bin-tolerance", type=float, default=4.0)
    parser.add_argument("--peak-width-db", type=float, default=-3.0)
    parser.add_argument("--max-files", type=int, default=1, help="Default is a small trial run; set 0 for all files.")
    parser.add_argument("--max-packets-per-file", type=int, default=3, help="Default is a small trial run; set 0 for no packet limit.")
    parser.add_argument("--max-scan-symbols", type=int, default=None)
    parser.add_argument("--sample-starts", default=None, help="Comma-separated explicit packet preamble starts.")
    parser.add_argument("--periodic-extract", dest="periodic_extract", action="store_true", default=True)
    parser.add_argument("--no-periodic-extract", dest="periodic_extract", action="store_false")
    parser.add_argument("--packet-period-samples", type=int, default=None)
    parser.add_argument("--packet-period-seconds", type=float, default=5.0)
    parser.add_argument("--period-search-samples", type=int, default=50_000)
    parser.add_argument("--max-period-misses", type=int, default=4)
    parser.add_argument("--seed-max-scan-symbols", type=int, default=None)

    parser.add_argument("--side-exclusion-bins", type=float, default=0.5)
    parser.add_argument("--secondary-exclusion-bins", type=float, default=0.25)
    parser.add_argument(
        "--min-packet-score-db",
        type=float,
        default=None,
        help="Drop extracted packets whose packet-level score_db_mean is below this threshold.",
    )
    parser.add_argument("--plot-packets-per-file", type=int, default=2)
    parser.add_argument("--plot-floor-db", type=float, default=-50.0)
    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_values = parse_q_values(args.q_values)
    packet_starts_by_file: Optional[dict[str, list[dict]]] = None
    if args.packet_starts_csv is not None:
        packet_starts_by_file = load_packet_starts_csv(args.packet_starts_csv)
        if args.input.is_file():
            input_files = [args.input]
        else:
            input_files = [
                args.input / file_name
                for file_name in sorted(packet_starts_by_file)
                if (args.input / file_name).exists()
            ]
    else:
        input_files = collect_input_files(args.input, args.glob)
    if args.max_files is not None and args.max_files > 0:
        input_files = input_files[: args.max_files]
    if not input_files:
        raise FileNotFoundError(f"No files matched {args.input} / {args.glob}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args.output_dir / "run_config.json", args, q_values, input_files)

    long_csv = args.output_dir / "subbin_spectrum_long.csv"
    symbol_csv = args.output_dir / "symbol_peak_summary.csv"
    packet_csv = args.output_dir / "packet_q_summary.csv"
    point_csv = args.output_dir / "point_q_summary.csv"
    long_writer, long_handle = open_writer(long_csv, LONG_HEADER)
    symbol_writer, symbol_handle = open_writer(symbol_csv, SYMBOL_SUMMARY_HEADER)
    packet_writer, packet_handle = open_writer(packet_csv, PACKET_SUMMARY_HEADER)

    counts = {"long_rows": 0, "symbol_rows": 0, "packet_rows": 0}
    try:
        for path in input_files:
            packet_start_rows = packet_starts_by_file.get(path.name, []) if packet_starts_by_file is not None else None
            long_count, symbol_count, packet_count = process_file(
                path,
                q_values,
                (long_writer, symbol_writer, packet_writer),
                args,
                packet_start_rows,
            )
            counts["long_rows"] += long_count
            counts["symbol_rows"] += symbol_count
            counts["packet_rows"] += packet_count
    finally:
        long_handle.close()
        symbol_handle.close()
        packet_handle.close()

    point_count = write_point_summary(packet_csv, point_csv)
    print(f"Wrote {point_count} point summaries to {point_csv}")
    print(
        "Wrote "
        f"{counts['long_rows']} spectrum rows, "
        f"{counts['symbol_rows']} symbol summaries, "
        f"{counts['packet_rows']} packet summaries to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
