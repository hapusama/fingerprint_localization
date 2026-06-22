from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


EPS = 1e-12
WIDEBAND_FILE_RE = re.compile(
    r"^(?P<experiment>[^_\-]+)[_\-](?P<corridor>[^_\-]+)[_\-](?P<location>[^_\-]+)(?:[_\-].*)?$"
)
OFFSET_RE = re.compile(r"preamble_fft_mag_bin_([+-]?\d+)$")
PHASE_OFFSET_RE = re.compile(r"preamble_fft_phase_bin_([+-]?\d+)$")
FEATURE_NORM_RE = re.compile(r"feature_norm_(\d+)$")


@dataclass
class ChirpSummary:
    file_name: str
    path: str
    experiment_id: str
    corridor_id: str
    location_key: str
    is_fail_file: bool
    segments: int
    trusted_segments: int
    mean_corr_score: float
    max_corr_score: float
    rho_chirp: float
    rho_chirp_std: float
    tau_rms_chirp_us: float
    tau_rms_chirp_us_std: float
    main_delay_us: float
    total_pdp_power: float


def clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def finite_or(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(float(value)) else float(fallback)


def find_column(df: pd.DataFrame, exact: Sequence[str] = (), startswith: Sequence[str] = (), contains: Sequence[str] = ()) -> str:
    for name in exact:
        if name in df.columns:
            return name
    for column in df.columns:
        text = str(column)
        if any(text.startswith(prefix) for prefix in startswith):
            return column
    for column in df.columns:
        text = str(column)
        if all(token in text for token in contains):
            return column
    raise KeyError(f"Could not find column, exact={exact}, startswith={startswith}, contains={contains}")


def read_location_distance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    state_col = find_column(df, startswith=("c_i",), contains=())
    chirp_flag_col = find_column(df, startswith=("o_i",), contains=())
    out = pd.DataFrame(
        {
            "corridor_id": df["corridor_id"].map(clean_key),
            "location_key": df["location_id"].map(clean_key),
            "position_key": df.get("position_key", df["location_id"].map(clean_key)),
            "state_code": pd.to_numeric(df[state_col], errors="coerce").fillna(2).astype(int),
            "distance_chirp_flag": pd.to_numeric(df[chirp_flag_col], errors="coerce").fillna(0).astype(int),
            "distance_m": pd.to_numeric(df["distance_m"], errors="coerce"),
            "distance_source": df.get("distance_source", ""),
        }
    )
    state_map = {1: "LOS", 0: "OLOS", 2: "NLOS"}
    out["state"] = out["state_code"].map(state_map).fillna("NLOS")
    return out


def collect_bin_files(root: Path, pattern: str) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".bin":
        files = [root]
    else:
        files = sorted(root.rglob(pattern))
    cleaned: list[Path] = []
    for path in files:
        parts = {part.lower() for part in path.parts}
        if "__macosx" in parts or path.name.startswith("._"):
            continue
        cleaned.append(path)
    return cleaned


def parse_wideband_meta(path: Path) -> dict[str, str]:
    match = WIDEBAND_FILE_RE.match(path.stem)
    if not match:
        return {"experiment_id": "", "corridor_id": "", "location_key": path.stem}
    groups = match.groupdict()
    return {
        "experiment_id": clean_key(groups["experiment"]),
        "corridor_id": clean_key(groups["corridor"]),
        "location_key": clean_key(groups["location"]),
    }


def make_lfm_template(fs: float, chirp_bw: float, chirp_duration: float) -> np.ndarray:
    n = max(1, int(round(fs * chirp_duration)))
    t = np.arange(n, dtype=np.float64) / fs
    sweep_rate = chirp_bw / chirp_duration
    phase = 2.0 * np.pi * ((-chirp_bw / 2.0) * t + 0.5 * sweep_rate * t * t)
    return (np.exp(1j * phase) * np.hanning(n)).astype(np.complex64)


def fft_correlate_valid(segment: np.ndarray, ref: np.ndarray) -> np.ndarray:
    valid_len = segment.size - ref.size + 1
    if valid_len <= 0:
        return np.empty(0, dtype=np.complex64)
    n_corr = segment.size + ref.size - 1
    n_fft = 1 << int(np.ceil(np.log2(n_corr)))
    kernel = np.conj(ref[::-1])
    corr = np.fft.ifft(np.fft.fft(segment, n_fft) * np.fft.fft(kernel, n_fft))
    return corr[ref.size - 1 : ref.size - 1 + valid_len]


def local_energy_valid(segment: np.ndarray, ref_len: int) -> np.ndarray:
    power = np.abs(segment).astype(np.float64) ** 2
    prefix = np.concatenate(([0.0], np.cumsum(power)))
    return prefix[ref_len:] - prefix[:-ref_len]


def analyze_chirp_file(path: Path, ref: np.ndarray, args: argparse.Namespace) -> ChirpSummary:
    meta = parse_wideband_meta(path)
    is_fail = "fail" in path.stem.lower() or any("fail" in part.lower() for part in path.parts)
    if is_fail:
        return ChirpSummary(path.name, str(path), **meta, is_fail_file=True, segments=0, trusted_segments=0,
                            mean_corr_score=0.0, max_corr_score=0.0, rho_chirp=0.0, rho_chirp_std=0.0,
                            tau_rms_chirp_us=0.0, tau_rms_chirp_us_std=0.0, main_delay_us=0.0, total_pdp_power=0.0)

    x = np.memmap(path, dtype=np.complex64, mode="r")
    fs = float(args.fs)
    period_len = int(round(fs * args.period))
    if period_len <= ref.size:
        raise ValueError("--period must be longer than --chirp-duration")
    max_segments = x.size // period_len
    if args.max_segments > 0:
        max_segments = min(max_segments, args.max_segments)

    ref_energy = float(np.sum(np.abs(ref) ** 2))
    pre = max(0, int(round(args.pre_delay_us * 1e-6 * fs)))
    post = max(1, int(round(args.max_delay_us * 1e-6 * fs)))
    main_half = max(1, int(round(args.main_window_us * 0.5e-6 * fs)))

    rho_values: list[float] = []
    tau_values: list[float] = []
    delay_values: list[float] = []
    corr_scores: list[float] = []
    powers: list[float] = []

    for seg_id in range(max_segments):
        start = seg_id * period_len
        stop = min(x.size, start + period_len + ref.size - 1)
        segment = np.asarray(x[start:stop], dtype=np.complex64)
        corr = fft_correlate_valid(segment, ref)
        if corr.size == 0:
            continue
        local_energy = local_energy_valid(segment, ref.size)
        valid = local_energy > max(float(np.max(local_energy)) * 1e-6, EPS)
        score = np.zeros(corr.size, dtype=np.float64)
        score[valid] = np.abs(corr[valid]) / np.sqrt(local_energy[valid] * ref_energy + EPS)
        detect_index = int(np.argmax(score))
        detect_score = float(score[detect_index])
        corr_scores.append(detect_score)
        if detect_score < args.corr_gate:
            continue

        lo = max(0, detect_index - pre)
        hi = min(corr.size, detect_index + post + 1)
        pdp = np.abs(corr[lo:hi]).astype(np.float64) ** 2
        total = float(np.sum(pdp))
        if total <= EPS:
            continue
        peak = int(np.argmax(pdp))
        main_lo = max(0, peak - main_half)
        main_hi = min(pdp.size, peak + main_half + 1)
        main_energy = float(np.sum(pdp[main_lo:main_hi]))
        rho_values.append(main_energy / total)

        delays_us = (np.arange(lo, hi, dtype=np.float64) - detect_index) / fs * 1.0e6
        weights = pdp / total
        mean_delay = float(np.sum(delays_us * weights))
        tau_rms = float(np.sqrt(np.sum(((delays_us - mean_delay) ** 2) * weights)))
        tau_values.append(tau_rms)
        delay_values.append(float(delays_us[peak]))
        powers.append(total)

    return ChirpSummary(
        file_name=path.name,
        path=str(path),
        **meta,
        is_fail_file=False,
        segments=int(max_segments),
        trusted_segments=len(rho_values),
        mean_corr_score=float(np.mean(corr_scores)) if corr_scores else 0.0,
        max_corr_score=float(np.max(corr_scores)) if corr_scores else 0.0,
        rho_chirp=float(np.mean(rho_values)) if rho_values else 0.0,
        rho_chirp_std=float(np.std(rho_values)) if rho_values else 0.0,
        tau_rms_chirp_us=float(np.mean(tau_values)) if tau_values else 0.0,
        tau_rms_chirp_us_std=float(np.std(tau_values)) if tau_values else 0.0,
        main_delay_us=float(np.mean(delay_values)) if delay_values else 0.0,
        total_pdp_power=float(np.mean(powers)) if powers else 0.0,
    )


def analyze_wideband(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    if args.reuse_wideband_csv is not None and args.reuse_wideband_csv.exists():
        print(f"[wideband] reusing {args.reuse_wideband_csv}")
        df = pd.read_csv(args.reuse_wideband_csv)
        if "location_key" in df.columns:
            df["location_key"] = df["location_key"].map(clean_key)
        if "corridor_id" in df.columns:
            df["corridor_id"] = df["corridor_id"].map(clean_key)
        return df
    files = collect_bin_files(root, args.wideband_glob)
    if args.max_files > 0:
        files = files[: args.max_files]
    ref = make_lfm_template(args.fs, args.chirp_bw, args.chirp_duration)
    rows = []
    for index, path in enumerate(files, start=1):
        print(f"[wideband] {index}/{len(files)} {path.name}")
        summary = analyze_chirp_file(path, ref, args)
        rows.append(summary.__dict__)
    return pd.DataFrame(rows)


def sorted_mag_columns(df: pd.DataFrame) -> list[str]:
    pairs = []
    for column in df.columns:
        match = OFFSET_RE.match(str(column))
        if match:
            pairs.append((int(match.group(1)), column))
    if not pairs:
        raise ValueError("No preamble_fft_mag_bin_* columns found.")
    return [column for _, column in sorted(pairs)]


def sorted_feature_norm_columns(df: pd.DataFrame) -> list[str]:
    pairs = []
    for column in df.columns:
        match = FEATURE_NORM_RE.match(str(column))
        if match:
            pairs.append((int(match.group(1)), column))
    if not pairs:
        raise ValueError("No feature_norm_* columns found.")
    return [column for _, column in sorted(pairs)]


def sorted_phase_columns(df: pd.DataFrame) -> list[str]:
    pairs = []
    for column in df.columns:
        match = PHASE_OFFSET_RE.match(str(column))
        if match:
            pairs.append((int(match.group(1)), column))
    if not pairs:
        raise ValueError("No preamble_fft_phase_bin_* columns found.")
    return [column for _, column in sorted(pairs)]


def ensure_feature_norm_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    try:
        return df, sorted_feature_norm_columns(df)
    except ValueError:
        out = df.copy()
        mag_cols = sorted_mag_columns(out)
        phase_cols = sorted_phase_columns(out)
        raw = out[[*mag_cols, *phase_cols]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        std = np.where(std <= EPS, 1.0, std)
        norm = (np.nan_to_num(raw, nan=mean) - mean) / std
        feature_cols = []
        for idx in range(norm.shape[1]):
            column = f"feature_norm_{idx:02d}"
            out[column] = norm[:, idx].astype(np.float32)
            feature_cols.append(column)
        return out, feature_cols


def build_s17_features(usrp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df, feature_cols = ensure_feature_norm_columns(usrp_df)
    mag_cols = sorted_mag_columns(df)
    if "preamble_fft_mag_bin_+0" not in df.columns:
        raise ValueError("preamble_fft_mag_bin_+0 is required for S17/S16 concentration")

    df["location_key"] = df["position_id"].map(clean_key)
    if {"s17_c_s", "s17_j_s"}.issubset(df.columns):
        df["s17_c_s"] = pd.to_numeric(df["s17_c_s"], errors="coerce")
        df["s17_j_s"] = pd.to_numeric(df["s17_j_s"], errors="coerce")
        center = pd.to_numeric(df["preamble_fft_mag_bin_+0"], errors="coerce").to_numpy(dtype=np.float64)
        center_energy = np.maximum(center, 0.0) ** 2
        df["s17_center_energy"] = center_energy
        df["s17_j_source"] = "packet_symbol_bin0_energy_cv"
        j_note = "J_S came from packet-internal preamble symbol bin0 energy CV exported by the extractor."
    else:
        mag = df[mag_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        energy = np.maximum(mag, 0.0) ** 2
        center = pd.to_numeric(df["preamble_fft_mag_bin_+0"], errors="coerce").to_numpy(dtype=np.float64)
        center_energy = np.maximum(center, 0.0) ** 2
        df["s17_c_s"] = center_energy / np.maximum(np.sum(energy, axis=1), EPS)
        df["s17_center_energy"] = center_energy
        grouped_energy = df.groupby("location_key")["s17_center_energy"]
        loc_mean = grouped_energy.transform("mean")
        loc_std = grouped_energy.transform(lambda s: float(np.std(s, ddof=0)))
        df["s17_j_s"] = loc_std / np.maximum(loc_mean, EPS)
        df["s17_j_source"] = "location_bin0_energy_cv"
        j_note = "The input CSV lacked packet-internal S17 fields, so J_S was approximated by per-location bin0 energy CV across packets."

    grouped = df.groupby("location_key", as_index=False).agg(
        corridor_id=("corridor_id", lambda s: clean_key(s.iloc[0])),
        s17_c_s_mean=("s17_c_s", "mean"),
        s17_c_s_std=("s17_c_s", "std"),
        s17_j_s=("s17_j_s", "mean"),
        s17_center_energy_mean=("s17_center_energy", "mean"),
        usrp_packet_count=("s17_c_s", "size"),
        preamble_power_db_mean=("preamble_avg_power_db", "mean"),
        peak_to_residual_db_mean=("preamble_peak_to_residual_db", "mean"),
    )
    grouped["s17_c_s_std"] = grouped["s17_c_s_std"].fillna(0.0)

    metadata = {
        "bin_count_available": len(mag_cols),
        "c_s_note": f"C_S uses the {len(mag_cols)} local FFT bins available in the CSV.",
        "j_s_note": j_note,
        "feature_columns": feature_cols,
    }
    return df, grouped, metadata


def build_rssi_stats(path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path is None or not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    usecols = [
        "location_id",
        "sf",
        "tp",
        "realtime_average_rssi",
        "average_rssi",
        "rssi_variance",
        "snr",
        "median_rssi",
        "mode_rssi",
        "residual",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in set(usecols))
    df["location_key"] = df["location_id"].map(clean_key)
    for column in usecols:
        if column in df.columns and column != "location_id":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = df.groupby("location_key", as_index=False).agg(
        mean_RSSI=("realtime_average_rssi", "mean"),
        var_RSSI=("rssi_variance", "mean"),
        mean_SNR=("snr", "mean"),
        median_RSSI=("median_rssi", "mean"),
        mode_RSSI=("mode_rssi", "mean"),
        rssi_sample_count=("realtime_average_rssi", "size"),
    )
    return df, grouped


def fit_linear_model(x: np.ndarray, y: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.shape[0] < x.shape[1]:
        return np.zeros(x.shape[1], dtype=np.float64) + fallback
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef.astype(np.float64)


def fit_rho_model(point_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    train = point_df[
        (point_df["chirp_available"] > 0)
        & np.isfinite(point_df["rho_chirp"])
        & (point_df["rho_chirp"] > 0)
        & np.isfinite(point_df["s17_c_s_mean"])
        & np.isfinite(point_df["s17_j_s"])
    ].copy()
    metadata = {"train_points": int(len(train)), "method": "logit_linear_s17"}
    if len(train) < 2:
        return np.array([0.0, 4.0, -2.0], dtype=np.float64), {**metadata, "fallback": True}
    y = np.clip(train["rho_chirp"].to_numpy(dtype=np.float64), 1e-4, 1.0 - 1e-4)
    target = np.log(y / (1.0 - y))
    x = np.column_stack(
        [
            np.ones(len(train), dtype=np.float64),
            train["s17_c_s_mean"].to_numpy(dtype=np.float64),
            train["s17_j_s"].to_numpy(dtype=np.float64),
        ]
    )
    coef = fit_linear_model(x, target)
    return coef, {**metadata, "fallback": False, "coef": coef.tolist()}


def add_tau_and_physics(point_df: pd.DataFrame, rssi_grouped: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = point_df.copy()
    coef, rho_meta = fit_rho_model(df)
    x_rho = np.column_stack(
        [
            np.ones(len(df), dtype=np.float64),
            pd.to_numeric(df["s17_c_s_mean"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            pd.to_numeric(df["s17_j_s"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        ]
    )
    df["rho_hat_s17"] = sigmoid(x_rho @ coef)
    df.loc[df["state"] == "NLOS", "rho_hat_s17"] = 0.0
    df["rho_final"] = np.where(df["chirp_available"] > 0, df["rho_chirp"], df["rho_hat_s17"])
    df["rho_final"] = pd.to_numeric(df["rho_final"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    valid_tau = df[(df["chirp_available"] > 0) & (df["tau_rms_chirp_us"] > 0)]
    olos_tau = valid_tau[valid_tau["state"] == "OLOS"]["tau_rms_chirp_us"]
    if len(olos_tau) == 0:
        olos_tau = valid_tau["tau_rms_chirp_us"]
    high_tau = float(np.quantile(olos_tau, 0.75)) if len(olos_tau) else 1.0
    tau_by_state = valid_tau.groupby("state")["tau_rms_chirp_us"].mean().to_dict()
    df["tau_d_us"] = df["tau_rms_chirp_us"]
    for idx, row in df.iterrows():
        if row["chirp_available"] > 0 and row["tau_rms_chirp_us"] > 0:
            continue
        if row["state"] == "NLOS":
            tau = high_tau
        else:
            tau = tau_by_state.get(row["state"], valid_tau["tau_rms_chirp_us"].mean() if len(valid_tau) else high_tau)
        df.at[idx, "tau_d_us"] = finite_or(tau, high_tau)
    df["tau_d_us"] = pd.to_numeric(df["tau_d_us"], errors="coerce").fillna(high_tau).clip(lower=0.0)
    tau_ref = float(np.median(valid_tau["tau_rms_chirp_us"])) if len(valid_tau) else float(np.median(df["tau_d_us"]))
    tau_ref = max(tau_ref, 1e-6)

    rssi_train = df.merge(rssi_grouped, on="location_key", how="inner", suffixes=("", "_rssi")) if not rssi_grouped.empty else pd.DataFrame()
    if len(rssi_train) >= 2:
        x_path = np.column_stack(
            [
                np.ones(len(rssi_train), dtype=np.float64),
                np.log10(np.maximum(rssi_train["distance_m"].to_numpy(dtype=np.float64), 1e-3)),
            ]
        )
        y_rssi = rssi_train["mean_RSSI"].to_numpy(dtype=np.float64)
        path_coef = fit_linear_model(x_path, y_rssi)
    else:
        path_coef = np.array([-80.0, -20.0], dtype=np.float64)
    x_all_path = np.column_stack([np.ones(len(df)), np.log10(np.maximum(df["distance_m"].to_numpy(dtype=np.float64), 1e-3))])
    df["mean_RSSI_phy"] = x_all_path @ path_coef
    df["median_RSSI_phy"] = df["mean_RSSI_phy"]
    df["mode_RSSI_phy"] = df["mean_RSSI_phy"]

    if len(rssi_train) >= 2:
        x_snr = np.column_stack([np.ones(len(rssi_train)), rssi_train["mean_RSSI"].to_numpy(dtype=np.float64)])
        snr_coef = fit_linear_model(x_snr, rssi_train["mean_SNR"].to_numpy(dtype=np.float64))
    else:
        snr_coef = np.array([0.0, 0.0], dtype=np.float64)
    df["SNR_phy"] = snr_coef[0] + snr_coef[1] * df["mean_RSSI_phy"]

    if len(rssi_train) >= 3:
        merged_var = rssi_train.copy()
        x_var = np.column_stack(
            [
                np.ones(len(merged_var)),
                1.0 - np.clip(merged_var["rho_final"].to_numpy(dtype=np.float64), 0.0, 1.0) ** 2,
                np.log1p(np.maximum(merged_var["tau_d_us"].to_numpy(dtype=np.float64), 0.0) / tau_ref),
            ]
        )
        var_coef = fit_linear_model(x_var, np.maximum(merged_var["var_RSSI"].to_numpy(dtype=np.float64), 0.0))
    else:
        var_coef = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    x_all_var = np.column_stack(
        [
            np.ones(len(df)),
            1.0 - np.clip(df["rho_final"].to_numpy(dtype=np.float64), 0.0, 1.0) ** 2,
            np.log1p(np.maximum(df["tau_d_us"].to_numpy(dtype=np.float64), 0.0) / tau_ref),
        ]
    )
    df["var_RSSI_phy"] = np.maximum(x_all_var @ var_coef, 0.0)
    df["RSSI_residual_phy"] = 0.0

    meta = {
        "rho_model": rho_meta,
        "tau_ref_us": tau_ref,
        "nlos_tau_prior_us": high_tau,
        "path_loss_model": {"mean_RSSI_phy": "a + b*log10(distance_m)", "coef": path_coef.tolist()},
        "snr_model": {"SNR_phy": "q0 + q1*mean_RSSI_phy", "coef": snr_coef.tolist()},
        "variance_model": {
            "var_RSSI_phy": "s0 + s1*(1-rho^2) + s2*log(1+tau_d/tau_ref)",
            "coef": var_coef.tolist(),
        },
    }
    return df, meta


def build_point_physics(distance_df: pd.DataFrame, chirp_df: pd.DataFrame, s17_grouped: pd.DataFrame, rssi_grouped: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    chirp_success = chirp_df[~chirp_df["is_fail_file"]].copy() if not chirp_df.empty else pd.DataFrame()
    if not chirp_success.empty:
        chirp_by_loc = chirp_success.groupby("location_key", as_index=False).agg(
            chirp_file_count=("file_name", "size"),
            trusted_segments=("trusted_segments", "sum"),
            rho_chirp=("rho_chirp", "mean"),
            tau_rms_chirp_us=("tau_rms_chirp_us", "mean"),
            mean_corr_score=("mean_corr_score", "mean"),
            max_corr_score=("max_corr_score", "max"),
        )
    else:
        chirp_by_loc = pd.DataFrame(columns=["location_key", "chirp_file_count", "trusted_segments", "rho_chirp", "tau_rms_chirp_us", "mean_corr_score", "max_corr_score"])

    point = distance_df.merge(chirp_by_loc, on="location_key", how="left")
    point = point.merge(s17_grouped, on="location_key", how="left", suffixes=("", "_usrp"))
    point = point.merge(rssi_grouped, on="location_key", how="left")
    for column in ["chirp_file_count", "trusted_segments", "rho_chirp", "tau_rms_chirp_us", "mean_corr_score", "max_corr_score"]:
        point[column] = pd.to_numeric(point[column], errors="coerce").fillna(0.0)
    point["chirp_available"] = ((point["trusted_segments"] > 0) & (point["rho_chirp"] > 0)).astype(int)
    point, physics_meta = add_tau_and_physics(point, rssi_grouped)
    return point, physics_meta


def build_residual_dataset(rssi_df: pd.DataFrame, point_df: pd.DataFrame) -> pd.DataFrame:
    if rssi_df.empty:
        return pd.DataFrame()
    phy_cols = ["SNR_phy", "mean_RSSI_phy", "median_RSSI_phy", "mode_RSSI_phy", "var_RSSI_phy", "RSSI_residual_phy", "rho_final", "tau_d_us", "state_code"]
    merged = rssi_df.merge(point_df[["location_key", *phy_cols]], on="location_key", how="inner")
    if merged.empty:
        return pd.DataFrame()
    actual_cols = ["snr", "realtime_average_rssi", "median_rssi", "mode_rssi", "rssi_variance", "residual"]
    phy_actual_cols = ["SNR_phy", "mean_RSSI_phy", "median_RSSI_phy", "mode_RSSI_phy", "var_RSSI_phy", "RSSI_residual_phy"]
    for a_col, p_col in zip(actual_cols, phy_actual_cols):
        merged[f"residual_{a_col}"] = pd.to_numeric(merged[a_col], errors="coerce") - pd.to_numeric(merged[p_col], errors="coerce")
    return merged.dropna(subset=[f"residual_{col}" for col in actual_cols]).copy()


def save_residual_pth(residual_df: pd.DataFrame, output_path: Path) -> None:
    if residual_df.empty:
        return
    actual_cols = ["snr", "realtime_average_rssi", "median_rssi", "mode_rssi", "rssi_variance", "residual"]
    residual_cols = [f"residual_{col}" for col in actual_cols]
    phy_cols = ["SNR_phy", "mean_RSSI_phy", "median_RSSI_phy", "mode_RSSI_phy", "var_RSSI_phy", "RSSI_residual_phy"]
    state = pd.get_dummies(residual_df["state_code"].astype(int), prefix="state")
    for code in [0, 1, 2]:
        col = f"state_{code}"
        if col not in state.columns:
            state[col] = 0
    condition = np.column_stack(
        [
            residual_df["rho_final"].to_numpy(dtype=np.float32),
            residual_df["tau_d_us"].to_numpy(dtype=np.float32),
            state[["state_0", "state_1", "state_2"]].to_numpy(dtype=np.float32),
        ]
    )
    labels = residual_df["location_key"].map(clean_key)
    label_int = pd.to_numeric(labels, errors="coerce").fillna(-1).astype(int).to_numpy()
    payload = {
        "residual": torch.tensor(residual_df[residual_cols].to_numpy(dtype=np.float32)),
        "x_real": torch.tensor(residual_df[actual_cols].to_numpy(dtype=np.float32)),
        "x_phy": torch.tensor(residual_df[phy_cols].to_numpy(dtype=np.float32)),
        "condition": torch.tensor(condition, dtype=torch.float32),
        "label": torch.tensor(label_int, dtype=torch.int64),
        "metadata": {
            "actual_columns": actual_cols,
            "residual_columns": residual_cols,
            "condition_columns": ["rho_final", "tau_d_us", "state_0_OLOS", "state_1_LOS", "state_2_NLOS"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def split_by_location(df: pd.DataFrame, label_col: str, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, group in df.groupby(label_col):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_idx.extend(idx.tolist())
            continue
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_test = min(n_test, len(idx) - 1)
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def localization_error(pred: Sequence[str], true: Sequence[str], point_df: pd.DataFrame) -> float:
    distance = point_df.set_index("location_key")["distance_m"].to_dict()
    errors = []
    for p, t in zip(pred, true):
        if p in distance and t in distance:
            errors.append(abs(float(distance[p]) - float(distance[t])))
    return float(np.mean(errors)) if errors else float("nan")


def run_usrp_matching(usrp_packet_df: pd.DataFrame, point_df: pd.DataFrame, args: argparse.Namespace) -> dict:
    df = usrp_packet_df.copy()
    feature_cols = sorted_feature_norm_columns(df)
    df["location_key"] = df["position_id"].map(clean_key)
    train_idx, test_idx = split_by_location(df, "location_key", args.test_frac, args.seed)
    train = df.loc[train_idx].copy()
    test = df.loc[test_idx].copy()
    if test.empty or train.empty:
        return {"error": "empty train/test split"}

    prototypes = train.groupby("location_key")[feature_cols].mean()
    point_lookup = point_df.set_index("location_key")
    candidate_labels = list(prototypes.index)
    proto_matrix = prototypes.to_numpy(dtype=np.float64)

    true = test["location_key"].tolist()
    pred_fingerprint: list[str] = []
    pred_physics: list[str] = []
    for _, row in test.iterrows():
        x = row[feature_cols].to_numpy(dtype=np.float64)
        d_fp = np.mean((proto_matrix - x[None, :]) ** 2, axis=1)
        pred_fingerprint.append(candidate_labels[int(np.argmin(d_fp))])

        sample_rho = float(row.get("rho_hat_packet", row.get("s17_c_s", 0.0)))
        rho_penalty = []
        tau_penalty = []
        for label in candidate_labels:
            if label in point_lookup.index:
                rho_penalty.append((sample_rho - float(point_lookup.at[label, "rho_final"])) ** 2)
                tau_penalty.append(float(point_lookup.at[label, "tau_d_us"]))
            else:
                rho_penalty.append(0.0)
                tau_penalty.append(0.0)
        rho_penalty_arr = np.asarray(rho_penalty, dtype=np.float64)
        if np.std(rho_penalty_arr) > 0:
            rho_penalty_arr = (rho_penalty_arr - rho_penalty_arr.mean()) / (rho_penalty_arr.std() + EPS)
        d_norm = d_fp
        if np.std(d_norm) > 0:
            d_norm = (d_norm - d_norm.mean()) / (d_norm.std() + EPS)
        score = d_norm + args.rho_match_weight * rho_penalty_arr
        pred_physics.append(candidate_labels[int(np.argmin(score))])

    true_arr = np.asarray(true)
    fp_arr = np.asarray(pred_fingerprint)
    phy_arr = np.asarray(pred_physics)
    return {
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "candidate_locations": int(len(candidate_labels)),
        "fingerprint_only": {
            "accuracy": float(np.mean(fp_arr == true_arr)),
            "mean_distance_error_m": localization_error(fp_arr.tolist(), true, point_df),
        },
        "physics_match": {
            "accuracy": float(np.mean(phy_arr == true_arr)),
            "mean_distance_error_m": localization_error(phy_arr.tolist(), true, point_df),
            "rho_match_weight": float(args.rho_match_weight),
        },
    }


def attach_packet_rho(usrp_packet_df: pd.DataFrame, point_df: pd.DataFrame) -> pd.DataFrame:
    df = usrp_packet_df.copy()
    coef, _ = fit_rho_model(point_df)
    x = np.column_stack(
        [
            np.ones(len(df)),
            df["s17_c_s"].fillna(0.0).to_numpy(dtype=np.float64),
            df["s17_j_s"].fillna(0.0).to_numpy(dtype=np.float64),
        ]
    )
    df["rho_hat_packet"] = np.asarray(sigmoid(x @ coef), dtype=np.float64)
    df.loc[df["location_key"].map(point_df.set_index("location_key")["state"].to_dict()) == "NLOS", "rho_hat_packet"] = 0.0
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 physics-guided fingerprint artifacts from RSSI, USRP and wideband chirp data.")
    parser.add_argument("--location-distance", type=Path, default=Path("model/v1/output/location_distance_54points.csv"))
    parser.add_argument("--wideband-root", type=Path, default=Path("../dong/data_analysis"))
    parser.add_argument("--wideband-glob", default="*.bin")
    parser.add_argument("--reuse-wideband-csv", type=Path, default=None)
    parser.add_argument("--usrp-csv", type=Path, default=Path("data/processedData/usrp_preamble_fft_s17_54loc_20pkt_nonorm_relative_8sym.csv"))
    parser.add_argument("--rssi-plus-csv", type=Path, default=Path("data/processedData/FLOOR3/all_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/v2/output"))
    parser.add_argument("--residual-pth", type=Path, default=Path("model/v2/input/v2_residual_gan_dataset.pth"))
    parser.add_argument("--fs", type=float, default=20e6)
    parser.add_argument("--chirp-bw", type=float, default=18e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--corr-gate", type=float, default=0.08)
    parser.add_argument("--pre-delay-us", type=float, default=1.0)
    parser.add_argument("--max-delay-us", type=float, default=8.0)
    parser.add_argument("--main-window-us", type=float, default=0.12)
    parser.add_argument("--max-segments", type=int, default=40)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho-match-weight", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distance_df = read_location_distance(args.location_distance)
    chirp_df = analyze_wideband(args.wideband_root, args)
    usrp_df = pd.read_csv(args.usrp_csv)
    usrp_packet_df, s17_grouped, s17_meta = build_s17_features(usrp_df)
    rssi_df, rssi_grouped = build_rssi_stats(args.rssi_plus_csv)
    point_df, physics_meta = build_point_physics(distance_df, chirp_df, s17_grouped, rssi_grouped)
    usrp_packet_df = attach_packet_rho(usrp_packet_df, point_df)
    residual_df = build_residual_dataset(rssi_df, point_df)
    save_residual_pth(residual_df, args.residual_pth)
    match_metrics = run_usrp_matching(usrp_packet_df, point_df, args)

    chirp_df.to_csv(args.output_dir / "v2_wideband_chirp_features.csv", index=False)
    s17_grouped.to_csv(args.output_dir / "v2_s17_point_features.csv", index=False)
    usrp_packet_df.to_csv(args.output_dir / "v2_usrp_packet_features.csv", index=False)
    point_df.to_csv(args.output_dir / "v2_point_physics.csv", index=False)
    residual_df.to_csv(args.output_dir / "v2_residual_training.csv", index=False)
    (args.output_dir / "v2_match_metrics.json").write_text(json.dumps(match_metrics, indent=2), encoding="utf-8")
    metadata = {
        "inputs": {
            "location_distance": str(args.location_distance),
            "wideband_root": str(args.wideband_root),
            "usrp_csv": str(args.usrp_csv),
            "rssi_plus_csv": str(args.rssi_plus_csv),
        },
        "s17": s17_meta,
        "physics": physics_meta,
        "outputs": {
            "point_physics": str(args.output_dir / "v2_point_physics.csv"),
            "residual_training": str(args.output_dir / "v2_residual_training.csv"),
            "residual_pth": str(args.residual_pth),
            "match_metrics": str(args.output_dir / "v2_match_metrics.json"),
        },
    }
    (args.output_dir / "v2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote point physics  -> {args.output_dir / 'v2_point_physics.csv'}")
    print(f"Wrote residual data  -> {args.output_dir / 'v2_residual_training.csv'}")
    print(f"Wrote GAN dataset    -> {args.residual_pth}")
    print(f"Wrote match metrics  -> {args.output_dir / 'v2_match_metrics.json'}")
    print(json.dumps(match_metrics, indent=2))


if __name__ == "__main__":
    main()
