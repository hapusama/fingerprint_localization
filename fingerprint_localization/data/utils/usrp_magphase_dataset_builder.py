from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch


EPS = 1e-8
MAG_PREFIX = "preamble_fft_mag_bin_"
PHASE_PREFIX = "preamble_fft_phase_bin_"


def offset_suffix(offset: int) -> str:
    return f"{offset:+d}"


def make_offsets(bin_count: int) -> list[int]:
    if bin_count <= 0:
        raise ValueError("--bin-count must be positive.")
    left = bin_count // 2
    return list(range(-left, bin_count - left))


def clean_int(value: object) -> int:
    if pd.isna(value):
        raise ValueError("empty integer value")
    text = str(value).strip()
    if not re.fullmatch(r"-?\d+(?:\.0+)?", text):
        raise ValueError(f"not an integer value: {value!r}")
    return int(float(text))


def feature_columns(offsets: Sequence[int]) -> tuple[list[str], list[str]]:
    mag_columns = [f"{MAG_PREFIX}{offset_suffix(offset)}" for offset in offsets]
    phase_columns = [f"{PHASE_PREFIX}{offset_suffix(offset)}" for offset in offsets]
    return mag_columns, phase_columns


def normalize(values: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    normalized = (values - mean) / std
    return normalized.astype(np.float32), {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real-USRP 2x16 magnitude/phase classifier dataset from extracted LoRa spectrum CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-pth", type=Path, default=Path("model/v1/input/usrp_magphase16_real_dataset.pth"))
    parser.add_argument("--processed-csv", type=Path, default=Path("data/processedData/usrp_magphase16_real_dataset.csv"))
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--bin-count", type=int, default=16)
    parser.add_argument("--label-column", default="position_id")
    parser.add_argument("--all-label-count", type=int, default=54)
    parser.add_argument("--min-score-db", type=float, default=0.0)
    parser.add_argument("--max-peak-std", type=float, default=2.0)
    parser.add_argument("--keep-raw", action="store_true", help="Also store raw 2x16 features in the .pth payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # bin_count=16 时 offsets 为 [-8, -7, ..., +7]。
    # 这两组列名必须和特征提取脚本输出的 CSV 对上。
    offsets = make_offsets(args.bin_count)
    mag_columns, phase_columns = feature_columns(offsets)

    df = pd.read_csv(args.input_csv)

    # 1) 检查 CSV 是否包含标签列、16 个幅度列、16 个相位列。
    #    标签默认来自文件名解析出来的 position_id。
    required = [args.label_column, *mag_columns, *phase_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2) 做一次质量过滤。
    #    peak_to_residual 越大，说明 dechirp+FFT 的峰越尖；
    #    peak_bin_std 越小，说明多个前导码 symbol 的峰位置越稳定。
    before = len(df)
    if "preamble_peak_to_residual_db" in df.columns:
        df = df[pd.to_numeric(df["preamble_peak_to_residual_db"], errors="coerce") >= args.min_score_db]
    if "preamble_peak_bin_std" in df.columns:
        df = df[pd.to_numeric(df["preamble_peak_bin_std"], errors="coerce") <= args.max_peak_std]
    df = df.copy()

    for column in [*mag_columns, *phase_columns]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[*mag_columns, *phase_columns, args.label_column]).copy()
    df["label"] = df[args.label_column].map(clean_int).astype(int)

    # 3) 组装 2x16 输入张量。
    #    channel 0: 16 个 bin 的幅度
    #    channel 1: 16 个 bin 的相位
    #    raw_2x16 的形状是 [N, 2, 16]。
    mag = df[mag_columns].to_numpy(dtype=np.float32)
    phase = df[phase_columns].to_numpy(dtype=np.float32)
    raw_2x16 = np.stack([mag, phase], axis=1)

    # 当前 MLP 分类器吃一维向量，所以再把 [2,16] 拉平成 32 维。
    # 保留 features_2x16 是为了后续换 CNN/扩散模型时还能恢复二维语义。
    raw_flat = raw_2x16.reshape(len(df), -1)

    # 4) 对每个特征维度做标准化，避免幅度和相位量纲不同影响训练。
    #    norm_2x16 仍然保持 [N, 2, 16]，norm_flat 是 [N, 32]。
    norm_flat, stats = normalize(raw_flat)
    norm_2x16 = norm_flat.reshape(len(df), 2, args.bin_count)

    norm_columns = [f"feature_norm_{idx:02d}" for idx in range(norm_flat.shape[1])]
    for idx, column in enumerate(norm_columns):
        df[column] = norm_flat[:, idx]

    label_counts = df["label"].value_counts().sort_index()
    present_labels = label_counts.index.astype(int).tolist()
    expected_labels = list(range(args.all_label_count))
    missing_labels = sorted(set(expected_labels) - set(present_labels))

    # 5) metadata 记录数据集的来源、形状、bin 顺序、归一化参数和缺失标签。
    #    后续复现实验时，先看 metadata 就能知道这批数据到底包含哪些点。
    metadata = {
        "source_csv": str(args.input_csv),
        "rows_in": int(before),
        "rows_out": int(len(df)),
        "shape_2x16": [int(len(df)), 2, int(args.bin_count)],
        "feature_order": {
            "channel_0": "magnitude",
            "channel_1": "relative_phase",
            "bin_offsets": offsets,
            "flat_order": "mag[-8..+7], phase[-8..+7] after row-major flatten of [2,16]",
        },
        "filters": {
            "min_score_db": float(args.min_score_db),
            "max_peak_std": float(args.max_peak_std),
        },
        "normalization": stats,
        "all_label_count": int(args.all_label_count),
        "present_labels": present_labels,
        "missing_labels": missing_labels,
        "label_counts": {str(int(k)): int(v) for k, v in label_counts.items()},
    }

    payload = {
        "features_2x16": torch.tensor(norm_2x16, dtype=torch.float32),
        "features_flat": torch.tensor(norm_flat, dtype=torch.float32),

        # 兼容旧代码里的命名习惯：原项目很多地方把输入特征叫 rssi。
        # 这里 rssi 实际上不是 RSSI，而是 32 维频谱幅相特征。
        "rssi": torch.tensor(norm_flat, dtype=torch.float32),
        "label": torch.tensor(df["label"].to_numpy(dtype=np.int64), dtype=torch.int64),
        "metadata": metadata,
    }
    if args.keep_raw:
        payload["features_2x16_raw"] = torch.tensor(raw_2x16, dtype=torch.float32)
        payload["features_flat_raw"] = torch.tensor(raw_flat, dtype=torch.float32)

    args.output_pth.parent.mkdir(parents=True, exist_ok=True)
    args.processed_csv.parent.mkdir(parents=True, exist_ok=True)

    # 6) 保存三份东西：
    #    .pth 给 PyTorch 训练用；
    #    processed CSV 方便人工检查；
    #    metadata JSON 记录实验配置和归一化参数。
    torch.save(payload, args.output_pth)
    df.to_csv(args.processed_csv, index=False)

    metadata_json = args.metadata_json or args.processed_csv.with_suffix(".metadata.json")
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Read {before} rows, kept {len(df)} valid rows.")
    print(f"Present labels: {len(present_labels)} / {args.all_label_count}")
    print(f"Missing labels: {missing_labels}")
    print(f"Wrote dataset  -> {args.output_pth}")
    print(f"Wrote CSV      -> {args.processed_csv}")
    print(f"Wrote metadata -> {metadata_json}")


if __name__ == "__main__":
    main()
