from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


EPS = 1e-8
MAG_PREFIXES = {
    "fft": "preamble_fft_mag_bin_",
    "peak": "preamble_peak_mag_bin_",
}
PHASE_PREFIX = "preamble_fft_phase_bin_"
SF_COLUMNS = ("filename_sf", "sf")
TP_COLUMNS = ("filename_tx_power_dbm", "tx_power_dbm", "tp")


def offset_suffix(offset: int) -> str:
    return f"{offset:+d}"


def make_offsets(bin_count: int) -> list[int]:
    if bin_count <= 0:
        raise ValueError("--bin-count must be positive.")
    left = bin_count // 2
    return list(range(-left, bin_count - left))


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def parse_optional_ints(values: Sequence[str] | None) -> set[int] | None:
    if not values:
        return None
    parsed: set[int] = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.add(int(part))
    return parsed


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str], name: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Could not find a {name} column. Tried: {', '.join(candidates)}")


def collect_input_files(inputs: Sequence[Path], glob_pattern: str) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            files.append(input_path)
        elif input_path.is_dir():
            files.extend(sorted(input_path.glob(glob_pattern)))
        else:
            raise FileNotFoundError(input_path)
    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError(f"No CSV files matched {inputs} / {glob_pattern}")
    return unique_files


def load_spectrum_tables(inputs: Sequence[Path], glob_pattern: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in collect_input_files(inputs, glob_pattern):
        frame = pd.read_csv(path)
        frame["source_csv"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def available_offsets(df: pd.DataFrame, prefix: str) -> set[int]:
    offsets: set[int] = set()
    pattern = re.compile(rf"^{re.escape(prefix)}([+-]\d+)$")
    for column in df.columns:
        match = pattern.match(column)
        if match:
            offsets.add(int(match.group(1)))
    return offsets


def choose_mag_prefix(df: pd.DataFrame, family: str) -> tuple[str, str]:
    if family != "auto":
        prefix = MAG_PREFIXES[family]
        if available_offsets(df, prefix):
            return family, prefix
        raise ValueError(f"No columns found for requested feature family: {family}")

    for name in ("fft", "peak"):
        prefix = MAG_PREFIXES[name]
        if available_offsets(df, prefix):
            return name, prefix
    raise ValueError("No spectrum magnitude columns were found.")


def select_feature_columns(
    df: pd.DataFrame,
    bin_count: int,
    feature_family: str,
    include_phase: str,
) -> tuple[list[str], dict]:
    offsets = make_offsets(bin_count)
    family_name, mag_prefix = choose_mag_prefix(df, feature_family)

    mag_available = available_offsets(df, mag_prefix)
    missing_mag = [offset for offset in offsets if offset not in mag_available]
    if missing_mag:
        raise ValueError(
            f"Missing magnitude bins for {family_name}: "
            f"{', '.join(offset_suffix(offset) for offset in missing_mag)}"
        )

    mag_columns = [f"{mag_prefix}{offset_suffix(offset)}" for offset in offsets]
    phase_available = available_offsets(df, PHASE_PREFIX)
    phase_ready = all(offset in phase_available for offset in offsets)

    if include_phase == "yes" and not phase_ready:
        missing = [offset for offset in offsets if offset not in phase_available]
        raise ValueError(
            "Phase columns were requested but missing bins: "
            f"{', '.join(offset_suffix(offset) for offset in missing)}"
        )

    use_phase = include_phase == "yes" or (include_phase == "auto" and phase_ready)
    phase_columns = [f"{PHASE_PREFIX}{offset_suffix(offset)}" for offset in offsets] if use_phase else []
    columns = mag_columns + phase_columns

    feature_info = {
        "feature_family": family_name,
        "bin_count": bin_count,
        "offsets": offsets,
        "magnitude_columns": mag_columns,
        "phase_columns": phase_columns,
        "feature_dim": len(columns),
    }
    return columns, feature_info


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prepare_labels(
    df: pd.DataFrame,
    location_vector_path: Path,
    label_column: str,
    label_mode: str,
    drop_invalid: bool,
) -> tuple[pd.DataFrame, str]:
    loc_df = pd.read_csv(location_vector_path)
    if "idx" not in loc_df.columns:
        raise ValueError(f"{location_vector_path} must contain an idx column.")
    valid_idx = set(pd.to_numeric(loc_df["idx"], errors="coerce").dropna().astype(int))

    label_text = df[label_column].map(clean_id)
    numeric_labels = pd.to_numeric(label_text, errors="coerce")
    numeric_valid = numeric_labels.notna()

    resolved_mode = label_mode
    if label_mode == "auto":
        numeric_set = set(numeric_labels[numeric_valid].astype(int))
        resolved_mode = "idx" if numeric_set and numeric_set.issubset(valid_idx) else "location_id"

    df = df.copy()
    before = len(df)

    if resolved_mode == "idx":
        df["label"] = numeric_labels
        invalid_mask = df["label"].isna()
        if invalid_mask.any():
            if not drop_invalid:
                bad = sorted(set(label_text[invalid_mask]))
                raise ValueError(f"Non-numeric labels found for idx mode: {bad}")
            df = df.loc[~invalid_mask].copy()
        df["label"] = df["label"].astype(int)
    else:
        if "location_id" not in loc_df.columns:
            raise ValueError(f"{location_vector_path} must contain location_id for location_id mode.")
        id_to_idx = {
            clean_id(row["location_id"]): int(row["idx"])
            for _, row in loc_df.iterrows()
            if pd.notna(row["idx"])
        }
        df["label"] = label_text.map(id_to_idx)
        missing_mask = df["label"].isna()
        if missing_mask.any():
            if not drop_invalid:
                bad = sorted(set(label_text[missing_mask]))
                raise ValueError(f"Labels missing from location vector: {bad[:20]}")
            df = df.loc[~missing_mask].copy()
        df["label"] = df["label"].astype(int)

    missing_idx = sorted(set(df["label"]) - valid_idx)
    if missing_idx:
        raise ValueError(
            f"Labels have no condition vectors in {location_vector_path}: {missing_idx[:20]}"
        )

    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows while preparing labels.")
    return df, resolved_mode


def add_normalized_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[pd.DataFrame, list[str], dict]:
    df = df.copy()
    coerce_numeric(df, feature_columns)
    before = len(df)
    df = df.dropna(subset=list(feature_columns)).copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing spectrum features.")

    values = df.loc[:, feature_columns].astype(float)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0).replace(0, 1.0)
    normalized = (values - means) / (stds + EPS)

    normalized_columns = [f"spectrum_feature_{idx:02d}" for idx in range(len(feature_columns))]
    df.loc[:, normalized_columns] = normalized.to_numpy(dtype=np.float32)

    stats = {
        "source_feature_columns": list(feature_columns),
        "normalized_feature_columns": normalized_columns,
        "mean": {column: float(means[column]) for column in feature_columns},
        "std": {column: float(stds[column]) for column in feature_columns},
    }
    return df, normalized_columns, stats


def filter_rows(
    df: pd.DataFrame,
    sf_column: str,
    tp_column: str,
    keep_sf: set[int] | None,
    keep_tp: set[int] | None,
    keep_labels: set[int] | None,
) -> pd.DataFrame:
    result = df
    if keep_sf is not None:
        result = result[result[sf_column].astype(int).isin(keep_sf)]
    if keep_tp is not None:
        result = result[result[tp_column].astype(int).isin(keep_tp)]
    if keep_labels is not None:
        result = result[result["label"].astype(int).isin(keep_labels)]
    return result.copy()


def split_group_indices(
    indices: np.ndarray,
    pretrain_frac: float,
    finetune_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    indices = np.array(indices, dtype=int)
    rng.shuffle(indices)
    n = len(indices)
    if n == 0:
        return [], [], []
    if n == 1:
        return indices.tolist(), [], []
    if n == 2:
        return [int(indices[0])], [], [int(indices[1])]

    frac_sum = pretrain_frac + finetune_frac + test_frac
    if frac_sum <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    finetune_frac = finetune_frac / frac_sum
    test_frac = test_frac / frac_sum

    n_test = max(1, int(round(n * test_frac)))
    n_finetune = max(1, int(round(n * finetune_frac)))
    if n_test + n_finetune >= n:
        overflow = n_test + n_finetune - (n - 1)
        reduce_test = min(overflow, max(0, n_test - 1))
        n_test -= reduce_test
        overflow -= reduce_test
        n_finetune = max(0, n_finetune - overflow)

    test_idx = indices[:n_test]
    finetune_idx = indices[n_test : n_test + n_finetune]
    pretrain_idx = indices[n_test + n_finetune :]
    return pretrain_idx.tolist(), finetune_idx.tolist(), test_idx.tolist()


def stratified_split(
    df: pd.DataFrame,
    pretrain_frac: float,
    finetune_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pretrain_indices: list[int] = []
    finetune_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in df.groupby("label", sort=True):
        pre_idx, ft_idx, te_idx = split_group_indices(
            group.index.to_numpy(),
            pretrain_frac,
            finetune_frac,
            test_frac,
            rng,
        )
        pretrain_indices.extend(pre_idx)
        finetune_indices.extend(ft_idx)
        test_indices.extend(te_idx)

    return (
        df.loc[pretrain_indices].sample(frac=1.0, random_state=seed).copy(),
        df.loc[finetune_indices].sample(frac=1.0, random_state=seed).copy(),
        df.loc[test_indices].sample(frac=1.0, random_state=seed).copy(),
    )


def cap_per_label(df: pd.DataFrame, max_per_label: int | None, seed: int) -> pd.DataFrame:
    if max_per_label is None or max_per_label <= 0 or df.empty:
        return df
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min(len(group), max_per_label), random_state=seed))
        .sample(frac=1.0, random_state=seed)
        .copy()
    )


def save_split(
    df: pd.DataFrame,
    path: Path,
    feature_columns: Sequence[str],
    sf_column: str,
    tp_column: str,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_tensor = torch.tensor(df.loc[:, feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    sf_values = df.loc[:, sf_column].to_numpy(dtype=np.float32)
    tp_values = df.loc[:, tp_column].to_numpy(dtype=np.float32)
    payload = {
        "rssi": feature_tensor,
        "sf": torch.tensor(sf_values, dtype=torch.float32) / 10.0,
        "tp": torch.tensor(tp_values, dtype=torch.float32) / 10.0,
        "snr": torch.tensor(np.stack([sf_values, tp_values], axis=1), dtype=torch.float32) / 10.0,
        "label": torch.tensor(df.loc[:, "label"].to_numpy(dtype=np.int64), dtype=torch.int64),
        "metadata": metadata,
    }
    torch.save(payload, path)
    print(f"Wrote {len(df):5d} rows -> {path}")


def save_finger(
    df: pd.DataFrame,
    path: Path,
    feature_columns: Sequence[str],
    sf_column: str,
    tp_column: str,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_values = df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    cond_values = df.loc[:, [sf_column, tp_column]].to_numpy(dtype=np.float32) / 10.0
    payload = {
        "features": torch.tensor(np.concatenate([feature_values, cond_values], axis=1), dtype=torch.float32),
        "label": torch.tensor(df.loc[:, "label"].to_numpy(dtype=np.int64), dtype=torch.int64),
        "metadata": metadata,
    }
    torch.save(payload, path)
    print(f"Wrote {len(df):5d} rows -> {path}")


def summarize_split(name: str, df: pd.DataFrame) -> None:
    labels = sorted(df["label"].astype(int).unique().tolist()) if not df.empty else []
    print(f"{name}: rows={len(df)}, labels={labels}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RGM-compatible .pth datasets from extracted LoRa spectrum CSV files."
    )
    parser.add_argument("--input", type=Path, nargs="+", default=[Path("data/packet_features_analysis")])
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--location-vector", type=Path, default=Path("model/v1/output/location_vector_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/v1/input"))
    parser.add_argument("--processed-csv", type=Path, default=Path("data/processedData/spectrum_features_all.csv"))
    parser.add_argument("--dataset-prefix", default="spectrum_features")
    parser.add_argument("--bin-count", type=int, default=16)
    parser.add_argument("--feature-family", choices=("auto", "fft", "peak"), default="auto")
    parser.add_argument("--include-phase", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--label-column", default="position_id")
    parser.add_argument("--label-mode", choices=("auto", "idx", "location_id"), default="auto")
    parser.add_argument("--drop-invalid-labels", action="store_true", default=True)
    parser.add_argument("--sf", nargs="*", default=None, help="Optional SF filter, e.g. --sf 11 12")
    parser.add_argument("--tp", nargs="*", default=None, help="Optional TX-power filter, e.g. --tp 2 6")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional label idx filter.")
    parser.add_argument("--pretrain-frac", type=float, default=0.70)
    parser.add_argument("--finetune-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-pretrain-per-label", type=int, default=None)
    parser.add_argument("--max-finetune-per-label", type=int, default=None)
    parser.add_argument("--max-test-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_spectrum_tables(args.input, args.glob)
    print(f"Loaded {len(df)} rows from extracted spectrum CSV files.")

    if args.label_column not in df.columns:
        raise ValueError(f"Input CSV files must contain {args.label_column}.")

    feature_columns, feature_info = select_feature_columns(
        df,
        bin_count=args.bin_count,
        feature_family=args.feature_family,
        include_phase=args.include_phase,
    )
    print(
        f"Using {feature_info['feature_family']} spectrum features: "
        f"{feature_info['feature_dim']} dims."
    )

    sf_column = first_existing_column(df, SF_COLUMNS, "SF")
    tp_column = first_existing_column(df, TP_COLUMNS, "TX power")
    coerce_numeric(df, [sf_column, tp_column])
    df = df.dropna(subset=[sf_column, tp_column]).copy()
    df[sf_column] = df[sf_column].astype(int)
    df[tp_column] = df[tp_column].astype(int)

    df, resolved_label_mode = prepare_labels(
        df,
        location_vector_path=args.location_vector,
        label_column=args.label_column,
        label_mode=args.label_mode,
        drop_invalid=args.drop_invalid_labels,
    )
    print(f"Using label mode: {resolved_label_mode}")

    df = filter_rows(
        df,
        sf_column=sf_column,
        tp_column=tp_column,
        keep_sf=parse_optional_ints(args.sf),
        keep_tp=parse_optional_ints(args.tp),
        keep_labels=parse_optional_ints(args.labels),
    )
    if df.empty:
        raise ValueError("No rows left after label and SF/TP filtering.")

    df, normalized_columns, norm_stats = add_normalized_features(df, feature_columns)
    if df.empty:
        raise ValueError("No rows left after feature cleanup.")

    metadata = {
        "dataset_prefix": args.dataset_prefix,
        "input": [str(path) for path in args.input],
        "location_vector": str(args.location_vector),
        "label_column": args.label_column,
        "label_mode": resolved_label_mode,
        "sf_column": sf_column,
        "tp_column": tp_column,
        "feature_info": feature_info,
        "normalization": norm_stats,
    }

    args.processed_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.processed_csv, index=False)
    metadata_path = args.processed_csv.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote processed CSV -> {args.processed_csv}")
    print(f"Wrote metadata      -> {metadata_path}")

    pretrain_df, finetune_df, test_df = stratified_split(
        df,
        pretrain_frac=args.pretrain_frac,
        finetune_frac=args.finetune_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    pretrain_df = cap_per_label(pretrain_df, args.max_pretrain_per_label, args.seed)
    finetune_df = cap_per_label(finetune_df, args.max_finetune_per_label, args.seed)
    test_df = cap_per_label(test_df, args.max_test_per_label, args.seed)

    summarize_split("pretrain", pretrain_df)
    summarize_split("finetune", finetune_df)
    summarize_split("test", test_df)

    save_split(
        pretrain_df,
        args.output_dir / f"{args.dataset_prefix}_pretrain_dataset.pth",
        normalized_columns,
        sf_column,
        tp_column,
        metadata,
    )
    save_split(
        finetune_df,
        args.output_dir / f"{args.dataset_prefix}_finetune_dataset.pth",
        normalized_columns,
        sf_column,
        tp_column,
        metadata,
    )
    save_split(
        test_df,
        args.output_dir / f"{args.dataset_prefix}_test_dataset.pth",
        normalized_columns,
        sf_column,
        tp_column,
        metadata,
    )
    save_finger(
        pretrain_df,
        args.output_dir / f"{args.dataset_prefix}_finger_dataset.pth",
        normalized_columns,
        sf_column,
        tp_column,
        metadata,
    )


if __name__ == "__main__":
    main()
