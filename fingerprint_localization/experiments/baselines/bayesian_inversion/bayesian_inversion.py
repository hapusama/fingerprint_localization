#!/usr/bin/env python3
"""Model-driven Bayesian inversion trial for LoRa fingerprint localization.

This is a pragmatic first implementation of the inverse-model idea:

    x_R, Y_q4 -> theta -> posterior over candidate locations

The q=4 likelihood is a forward-projection prototype likelihood with packet
nuisance minimization (sub-bin shift plus affine shape scaling). It intentionally
does not use v2 model outputs or v2_output_wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / "fingerprint_localization" / "data" / "mainline_202607"
DEFAULT_RSSI_CSV = DATA_ROOT / "inputs" / "rssi_plus_packet_level_54points.csv"
DEFAULT_LORA_FEATURE_CSV = DATA_ROOT / "inputs" / "lora_frequency_s17_54points.csv"
DEFAULT_CHIRP_CSV = DATA_ROOT / "features" / "chirp_point_multipath_structure_features.csv"
DEFAULT_Q4_SPECTRUM_CSV = DATA_ROOT / "external" / "subbin_spectrum_long.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

EPS = 1e-12
RSSI_PLUS_COLUMNS = [
    "snr",
    "realtime_average_rssi",
    "median_rssi",
    "mode_rssi",
    "rssi_variance",
    "residual",
]
RAW_OFFSETS = [-2.0, -1.0, 0.0, 1.0, 2.0]
THETA_COLUMNS = ["P0", "Psec", "tau_rms", "K_eff", "eta_diff", "alpha_asym"]
PacketKey = Tuple[str, int]


@dataclass
class PacketSample:
    key: PacketKey
    file_name: str
    packet_index: int
    label: str
    rssi_plus: List[float]
    log_a0: float
    q4_curve: List[float]
    q4_stability: float
    theta: List[float]


def file_stem(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_int(value: object) -> int:
    return int(float(value))


def point_label(corridor_id: object, position_id: object) -> str:
    return f"{parse_int(corridor_id)}_{parse_int(position_id)}"


def natural_label_key(label: str) -> Tuple[int, int]:
    corridor, position = label.split("_", 1)
    return int(corridor), int(position)


def point_display(label: str) -> str:
    corridor, position = label.split("_", 1)
    return f"c{corridor}p{position}"


def quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def iqr(values: Sequence[float]) -> float:
    return quantile(values, 0.75) - quantile(values, 0.25)


def safe_iqr(values: Sequence[float]) -> float:
    spread = iqr(values)
    return spread if spread > EPS else 1.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson_abs(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return 0.0
    mx = mean(x)
    my = mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= EPS or vy <= EPS:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return abs(cov / math.sqrt(vx * vy))


def read_csv_dict(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rssi_packets(path: Path) -> Dict[PacketKey, dict]:
    packets: Dict[PacketKey, dict] = {}
    for row in read_csv_dict(path):
        key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
        packets[key] = {
            "file_name": row["file_name"],
            "packet_index": key[1],
            "label": row["position_key"],
            "rssi_plus": [parse_float(row[col]) for col in RSSI_PLUS_COLUMNS],
        }
    return packets


def read_lora_energy(path: Path) -> Dict[PacketKey, dict]:
    packets: Dict[PacketKey, dict] = {}
    for row in read_csv_dict(path):
        key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
        label = row.get("position_key") or point_label(row["corridor_id"], row["position_id"])
        center = parse_float(row["preamble_fft_mag_bin_+0"])
        packets[key] = {
            "file_name": row["file_name"],
            "packet_index": key[1],
            "label": label,
            "log_a0": math.log(center + EPS),
        }
    return packets


def read_q4_packets(path: Path) -> Tuple[Dict[PacketKey, dict], List[float]]:
    symbols_by_packet: Dict[PacketKey, Dict[int, Dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    meta: Dict[PacketKey, dict] = {}
    offsets_seen = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if parse_int(row["q"]) != 4:
                continue
            offset = round(parse_float(row["subbin_offset"]), 6)
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
            symbol_id = parse_int(row.get("local_symbol_index", row.get("preamble_symbol_index", 0)))
            symbols_by_packet[key][symbol_id][offset] = parse_float(row["mag_db_rel_peak"])
            offsets_seen.add(offset)
            meta[key] = {
                "file_name": row["file_name"],
                "packet_index": key[1],
                "label": point_label(row["corridor_id"], row["position_id"]),
            }

    offsets = sorted(offsets_seen)
    packets: Dict[PacketKey, dict] = {}
    for key, symbols in symbols_by_packet.items():
        complete = [bins for bins in symbols.values() if all(offset in bins for offset in offsets)]
        if not complete:
            continue
        curve = [median([bins[offset] for bins in complete]) for offset in offsets]
        stability = median([iqr([bins[offset] for bins in complete]) for offset in offsets])
        packets[key] = {
            **meta[key],
            "q4_curve": curve,
            "q4_stability": stability,
            "q4_symbol_count": len(complete),
        }
    return packets, offsets


def theta_from_q4_curve(curve_db: Sequence[float], offsets: Sequence[float]) -> List[float]:
    powers = [10.0 ** (db / 10.0) for db in curve_db]
    total = sum(powers) + EPS
    weights = [p / total for p in powers]
    center_idx = min(range(len(offsets)), key=lambda i: abs(offsets[i]))
    center_power = powers[center_idx] / total
    side_power = max(0.0, 1.0 - center_power)
    tau_rms = math.sqrt(sum(w * (offset ** 2) for w, offset in zip(weights, offsets)))
    entropy = -sum(w * math.log(w + EPS) for w in weights)
    k_eff = math.exp(entropy)
    eta_diff = sum(w for w, offset in zip(weights, offsets) if abs(offset) > 0.5)
    left = sum(w for w, offset in zip(weights, offsets) if offset < 0.0)
    right = sum(w for w, offset in zip(weights, offsets) if offset > 0.0)
    alpha_asym = (right - left) / (right + left + EPS)
    return [
        math.log(center_power + EPS),
        math.log(side_power + EPS),
        tau_rms,
        k_eff,
        eta_diff,
        alpha_asym,
    ]


def align_samples(
    rssi_packets: Dict[PacketKey, dict],
    lora_packets: Dict[PacketKey, dict],
    q4_packets: Dict[PacketKey, dict],
    offsets: Sequence[float],
) -> List[PacketSample]:
    common = sorted(
        set(rssi_packets) & set(lora_packets) & set(q4_packets),
        key=lambda key: (natural_label_key(rssi_packets[key]["label"]), key[1], key[0]),
    )
    samples: List[PacketSample] = []
    for key in common:
        labels = {rssi_packets[key]["label"], lora_packets[key]["label"], q4_packets[key]["label"]}
        if len(labels) != 1:
            raise ValueError(f"Label mismatch for {key}: {labels}")
        q4_curve = q4_packets[key]["q4_curve"]
        samples.append(
            PacketSample(
                key=key,
                file_name=q4_packets[key]["file_name"],
                packet_index=key[1],
                label=rssi_packets[key]["label"],
                rssi_plus=rssi_packets[key]["rssi_plus"],
                log_a0=lora_packets[key]["log_a0"],
                q4_curve=q4_curve,
                q4_stability=q4_packets[key]["q4_stability"],
                theta=theta_from_q4_curve(q4_curve, offsets),
            )
        )
    return samples


def robust_point_prototypes(
    values: Sequence[Sequence[float]],
    labels: Sequence[str],
    train_indices: Sequence[int],
) -> Dict[str, dict]:
    grouped: Dict[str, List[Sequence[float]]] = defaultdict(list)
    for idx in train_indices:
        grouped[labels[idx]].append(values[idx])
    out: Dict[str, dict] = {}
    for label, rows in grouped.items():
        dim = len(rows[0])
        out[label] = {
            "median": [median([row[j] for row in rows]) for j in range(dim)],
            "iqr": [safe_iqr([row[j] for row in rows]) for j in range(dim)],
            "count": len(rows),
        }
    return out


def robust_scalar_prototypes(values: Sequence[float], labels: Sequence[str], train_indices: Sequence[int]) -> Dict[str, dict]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for idx in train_indices:
        grouped[labels[idx]].append(values[idx])
    return {
        label: {"median": median(rows), "iqr": safe_iqr(rows), "count": len(rows)}
        for label, rows in grouped.items()
    }


def robust_log_likelihood(x: Sequence[float], prototype: dict, weights: Sequence[float] | None = None) -> float:
    med = prototype["median"]
    spread = prototype["iqr"]
    if weights is None:
        weights = [1.0 / len(x)] * len(x)
    return -sum(weights[j] * ((x[j] - med[j]) / (spread[j] + EPS)) ** 2 for j in range(len(x)))


def scalar_log_likelihood(x: float, prototype: dict) -> float:
    return -((x - prototype["median"]) / (prototype["iqr"] + EPS)) ** 2


def interpolate_curve(curve: Sequence[float], offsets: Sequence[float], x: float) -> float | None:
    if x < offsets[0] or x > offsets[-1]:
        return None
    for idx in range(len(offsets) - 1):
        left = offsets[idx]
        right = offsets[idx + 1]
        if left <= x <= right:
            if abs(x - left) <= EPS:
                return curve[idx]
            if abs(x - right) <= EPS:
                return curve[idx + 1]
            frac = (x - left) / (right - left)
            return curve[idx] * (1.0 - frac) + curve[idx + 1] * frac
    return curve[-1]


def affine_fit_error(observed: Sequence[float], model: Sequence[float], allow_scale: bool) -> Tuple[float, float, float]:
    if len(observed) < 3 or len(observed) != len(model):
        return float("inf"), 0.0, 1.0
    mean_obs = mean(observed)
    mean_model = mean(model)
    if allow_scale:
        var_model = sum((v - mean_model) ** 2 for v in model)
        cov = sum((m - mean_model) * (o - mean_obs) for m, o in zip(model, observed))
        scale = cov / (var_model + EPS)
        scale = min(max(scale, 0.2), 2.5)
    else:
        scale = 1.0
    offset = mean_obs - scale * mean_model
    mse = sum((o - (offset + scale * m)) ** 2 for o, m in zip(observed, model)) / len(observed)
    return mse, offset, scale


def q4_forward_log_likelihood(
    observed: Sequence[float],
    prototype: Sequence[float],
    offsets: Sequence[float],
    shifts: Sequence[float],
    allow_scale: bool,
) -> Tuple[float, dict]:
    best = {"mse": float("inf"), "shift": 0.0, "offset": 0.0, "scale": 1.0}
    for shift in shifts:
        obs_valid: List[float] = []
        model_valid: List[float] = []
        for obs, offset in zip(observed, offsets):
            model_value = interpolate_curve(prototype, offsets, offset - shift)
            if model_value is None:
                continue
            obs_valid.append(obs)
            model_valid.append(model_value)
        mse, affine_offset, scale = affine_fit_error(obs_valid, model_valid, allow_scale)
        if mse < best["mse"]:
            best = {"mse": mse, "shift": shift, "offset": affine_offset, "scale": scale}
    return -best["mse"], best


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo <= EPS:
        return {label: 0.0 for label in scores}
    return {label: (value - lo) / (hi - lo) for label, value in scores.items()}


def read_chirp_theta(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}
    anchors: Dict[str, List[float]] = {}
    for row in read_csv_dict(path):
        label = row.get("position_key") or point_label(row["corridor_id"], row["location_id"])
        main_fraction = parse_float(row.get("main_effective_power_fraction"), 0.0)
        secondary_power = parse_float(row.get("secondary_effective_power_sum"), 0.0)
        tau_rms = parse_float(row.get("equivalent_rms_delay_us"), 0.0)
        k_eff = parse_float(row.get("effective_path_number"), 1.0)
        eta_diff = parse_float(row.get("unstable_secondary_peak_load"), 0.0)
        precursor = parse_float(row.get("precursor_effective_power"), 0.0)
        postcursor = parse_float(row.get("postcursor_effective_power"), 0.0)
        alpha_asym = (postcursor - precursor) / (postcursor + precursor + EPS)
        anchors[label] = [
            math.log(main_fraction + EPS),
            math.log(secondary_power + EPS),
            tau_rms,
            k_eff,
            eta_diff,
            alpha_asym,
        ]
    return anchors


def full_point_theta(samples: Sequence[PacketSample]) -> Dict[str, List[float]]:
    grouped: Dict[str, List[List[float]]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample.theta)
    out: Dict[str, List[float]] = {}
    for label, rows in grouped.items():
        out[label] = [median([row[j] for row in rows]) for j in range(len(THETA_COLUMNS))]
    return out


def calibrate_theta_weights(point_theta: Dict[str, List[float]], chirp_theta: Dict[str, List[float]], min_weight: float) -> Tuple[List[float], dict]:
    labels = sorted(set(point_theta) & set(chirp_theta), key=natural_label_key)
    raw_weights: List[float] = []
    correlations: Dict[str, float] = {}
    for j, name in enumerate(THETA_COLUMNS):
        x = [point_theta[label][j] for label in labels]
        y = [chirp_theta[label][j] for label in labels]
        corr = pearson_abs(x, y)
        correlations[name] = corr
        raw_weights.append(max(min_weight, corr))
    total = sum(raw_weights) or 1.0
    weights = [value / total for value in raw_weights]
    payload = {
        "anchor_count": len(labels),
        "anchor_labels": labels,
        "correlations_abs": correlations,
        "theta_weights": {name: weights[j] for j, name in enumerate(THETA_COLUMNS)},
        "min_weight": min_weight,
    }
    return weights, payload


def build_point_outputs(samples: Sequence[PacketSample], offsets: Sequence[float]) -> Tuple[List[dict], List[dict]]:
    packet_rows = []
    for sample in samples:
        row = {
            "file_name": sample.file_name,
            "packet_index": sample.packet_index,
            "position_key": sample.label,
            "q4_stability": sample.q4_stability,
            "log_a0": sample.log_a0,
        }
        row.update({name: sample.theta[j] for j, name in enumerate(THETA_COLUMNS)})
        row.update({f"q4_db_{offset:+.2f}": sample.q4_curve[j] for j, offset in enumerate(offsets)})
        packet_rows.append(row)

    grouped: Dict[str, List[PacketSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    point_rows = []
    for label in sorted(grouped, key=natural_label_key):
        rows = grouped[label]
        out = {"position_key": label, "display": point_display(label), "packet_count": len(rows)}
        for j, name in enumerate(THETA_COLUMNS):
            values = [sample.theta[j] for sample in rows]
            out[f"theta_{name}_median"] = median(values)
            out[f"theta_{name}_iqr"] = iqr(values)
        point_rows.append(out)
    return packet_rows, point_rows


def parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def evaluate(samples: Sequence[PacketSample], offsets: Sequence[float], theta_weights: Sequence[float], args: argparse.Namespace) -> Tuple[dict, List[dict], List[dict]]:
    labels = [sample.label for sample in samples]
    counts = Counter(labels)
    eval_indices = [idx for idx, label in enumerate(labels) if counts[label] >= 2]
    rssi_values = [sample.rssi_plus for sample in samples]
    energy_values = [sample.log_a0 for sample in samples]
    theta_values = [sample.theta for sample in samples]
    q4_values = [sample.q4_curve for sample in samples]
    shifts = parse_float_list(args.q4_shift_grid)

    prediction_rows: List[dict] = []
    candidate_rows: List[dict] = []
    rssi_correct = posterior_correct = topk_contains = 0

    for test_index in eval_indices:
        sample = samples[test_index]
        train_indices = [idx for idx in eval_indices if idx != test_index]
        rssi_proto = robust_point_prototypes(rssi_values, labels, train_indices)
        energy_proto = robust_scalar_prototypes(energy_values, labels, train_indices)
        theta_proto = robust_point_prototypes(theta_values, labels, train_indices)
        q4_proto = robust_point_prototypes(q4_values, labels, train_indices)

        rssi_scores = {
            label: robust_log_likelihood(sample.rssi_plus, proto)
            for label, proto in rssi_proto.items()
        }
        rssi_ranked = sorted(rssi_scores, key=lambda label: (-rssi_scores[label], natural_label_key(label)))
        candidates = rssi_ranked[: args.top_k]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        rssi_ok = int(rssi_pred == sample.label)
        rssi_correct += rssi_ok
        topk_contains += int(sample.label in candidates)

        lr_norm = normalize_scores({label: rssi_scores[label] for label in candidates})
        le_scores = {label: scalar_log_likelihood(sample.log_a0, energy_proto[label]) for label in candidates}
        le_norm = normalize_scores(le_scores)
        ltheta_scores = {
            label: robust_log_likelihood(sample.theta, theta_proto[label], theta_weights)
            for label in candidates
        }
        ltheta_norm = normalize_scores(ltheta_scores)

        lq_raw: Dict[str, float] = {}
        nuisance_by_label: Dict[str, dict] = {}
        for label in candidates:
            score, nuisance = q4_forward_log_likelihood(
                sample.q4_curve,
                q4_proto[label]["median"],
                offsets,
                shifts,
                args.fit_q4_affine_scale,
            )
            lq_raw[label] = score
            nuisance_by_label[label] = nuisance
        lq_norm = normalize_scores(lq_raw)

        posterior = {}
        for label in candidates:
            posterior[label] = (
                args.rssi_weight * lr_norm.get(label, 0.0)
                + args.beta * le_norm.get(label, 0.0)
                + args.gamma * lq_norm.get(label, 0.0)
                + args.delta * ltheta_norm.get(label, 0.0)
            )
            candidate_rows.append(
                {
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                    "candidate_label": label,
                    "candidate_display": point_display(label),
                    "posterior_score": posterior[label],
                    "L_R_norm": lr_norm.get(label, 0.0),
                    "L_E_norm": le_norm.get(label, 0.0),
                    "L_Q_norm": lq_norm.get(label, 0.0),
                    "L_theta_norm": ltheta_norm.get(label, 0.0),
                    "L_R_raw": rssi_scores[label],
                    "L_E_raw": le_scores[label],
                    "L_Q_raw": lq_raw[label],
                    "L_theta_raw": ltheta_scores[label],
                    "q4_shift": nuisance_by_label[label]["shift"],
                    "q4_affine_offset": nuisance_by_label[label]["offset"],
                    "q4_affine_scale": nuisance_by_label[label]["scale"],
                }
            )

        posterior_ranked = sorted(posterior, key=lambda label: (-posterior[label], natural_label_key(label)))
        pred = posterior_ranked[0]
        posterior_ok = int(pred == sample.label)
        posterior_correct += posterior_ok
        prediction_rows.append(
            {
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "true_display": point_display(sample.label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_display": point_display(rssi_pred),
                "rssi_top1_correct": rssi_ok,
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "posterior_label": pred,
                "posterior_display": point_display(pred),
                "posterior_correct": posterior_ok,
                "posterior_candidates": ";".join(posterior_ranked),
                "posterior_scores": ";".join(f"{label}:{posterior[label]:.6g}" for label in posterior_ranked),
                "q4_stability": sample.q4_stability,
            }
        )

    n = len(prediction_rows)
    metrics = {
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "rssi_top1_correct": rssi_correct,
        "rssi_top1_accuracy": rssi_correct / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n if n else 0.0,
        "posterior_correct": posterior_correct,
        "posterior_accuracy": posterior_correct / n if n else 0.0,
        "posterior_gain_vs_rssi_top1": posterior_correct - rssi_correct,
        "rssi_weight": args.rssi_weight,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "q4_shift_grid": shifts,
        "fit_q4_affine_scale": bool(args.fit_q4_affine_scale),
    }
    return metrics, prediction_rows, candidate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--lora-feature-csv", type=Path, default=DEFAULT_LORA_FEATURE_CSV)
    parser.add_argument("--q4-spectrum-csv", type=Path, default=DEFAULT_Q4_SPECTRUM_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=DEFAULT_CHIRP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.10, help="Energy likelihood weight.")
    parser.add_argument("--gamma", type=float, default=0.10, help="q=4 forward projection likelihood weight.")
    parser.add_argument("--delta", type=float, default=0.10, help="Inverted theta likelihood weight.")
    parser.add_argument("--theta-min-weight", type=float, default=0.10)
    parser.add_argument("--q4-shift-grid", default="-0.5,-0.25,0,0.25,0.5")
    parser.add_argument("--fit-q4-affine-scale", action="store_true", default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rssi_packets = read_rssi_packets(args.rssi_csv)
    lora_packets = read_lora_energy(args.lora_feature_csv)
    q4_packets, q4_offsets = read_q4_packets(args.q4_spectrum_csv)
    samples = align_samples(rssi_packets, lora_packets, q4_packets, q4_offsets)
    if not samples:
        raise RuntimeError("No aligned RSSI/LoRa/q4 packet samples found.")

    packet_rows, point_rows = build_point_outputs(samples, q4_offsets)
    theta_packet_fields = ["file_name", "packet_index", "position_key", "q4_stability", "log_a0"]
    theta_packet_fields += THETA_COLUMNS
    theta_packet_fields += [f"q4_db_{offset:+.2f}" for offset in q4_offsets]
    write_csv(args.output_dir / "theta_packet_features.csv", packet_rows, theta_packet_fields)

    theta_point_fields = ["position_key", "display", "packet_count"]
    for name in THETA_COLUMNS:
        theta_point_fields += [f"theta_{name}_median", f"theta_{name}_iqr"]
    write_csv(args.output_dir / "theta_point_prototypes.csv", point_rows, theta_point_fields)

    point_theta = full_point_theta(samples)
    chirp_theta = read_chirp_theta(args.chirp_structure_csv)
    theta_weights, calibration_payload = calibrate_theta_weights(point_theta, chirp_theta, args.theta_min_weight)
    with (args.output_dir / "theta_chirp_weight_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calibration_payload, f, indent=2, ensure_ascii=False)

    metrics, prediction_rows, candidate_rows = evaluate(samples, q4_offsets, theta_weights, args)
    prediction_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "true_display",
        "rssi_top1_label",
        "rssi_top1_display",
        "rssi_top1_correct",
        "rssi_topk_candidates",
        "true_in_rssi_topk",
        "posterior_label",
        "posterior_display",
        "posterior_correct",
        "posterior_candidates",
        "posterior_scores",
        "q4_stability",
    ]
    write_csv(args.output_dir / "v4_predictions.csv", prediction_rows, prediction_fields)

    candidate_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "candidate_label",
        "candidate_display",
        "posterior_score",
        "L_R_norm",
        "L_E_norm",
        "L_Q_norm",
        "L_theta_norm",
        "L_R_raw",
        "L_E_raw",
        "L_Q_raw",
        "L_theta_raw",
        "q4_shift",
        "q4_affine_offset",
        "q4_affine_scale",
    ]
    write_csv(args.output_dir / "v4_candidate_scores.csv", candidate_rows, candidate_fields)

    summary_fields = [
        "packet_count",
        "location_count",
        "top_k",
        "rssi_top1_correct",
        "rssi_top1_accuracy",
        "rssi_topk_contains_true",
        "rssi_topk_recall",
        "posterior_correct",
        "posterior_accuracy",
        "posterior_gain_vs_rssi_top1",
        "rssi_weight",
        "beta",
        "gamma",
        "delta",
        "q4_shift_grid",
        "fit_q4_affine_scale",
    ]
    write_csv(args.output_dir / "v4_summary.csv", [metrics], summary_fields)

    payload = {
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "lora_feature_csv": str(args.lora_feature_csv),
            "q4_spectrum_csv": str(args.q4_spectrum_csv),
            "chirp_structure_csv": str(args.chirp_structure_csv),
            "aligned_packet_count": len(samples),
            "aligned_location_count": len(Counter(sample.label for sample in samples)),
        },
        "data_policy": "uses curated files under data/mainline_202607; v2_output_wrong is not consumed",
        "method": {
            "name": "model-driven Bayesian inversion",
            "theta": THETA_COLUMNS,
            "candidate_stage": "RSSI+ robust Bayesian Top-K",
            "q4_likelihood": "forward q4 prototype likelihood with nuisance minimization over sub-bin shift and affine shape scale",
            "posterior": "L_R + beta*L_E + gamma*L_Q + delta*L_theta",
            "chirp_calibration": "chirp anchor correlations calibrate theta dimension weights",
        },
        "theta_weight_calibration": calibration_payload,
        "metrics": metrics,
    }
    with (args.output_dir / "v4_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
