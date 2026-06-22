from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v2 physics/GAN artifacts for quick experiment auditing.")
    parser.add_argument("--output-dir", type=Path, default=Path("model/v2/output"))
    parser.add_argument("--residual-pth", type=Path, default=Path("model/v2/input/v2_residual_gan_dataset.pth"))
    parser.add_argument("--s17-csv", type=Path, default=Path("data/processedData/usrp_preamble_fft_s17_54loc_20pkt_nonorm_relative_8sym.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("model/v2/output/v2_experiment_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point = pd.read_csv(args.output_dir / "v2_point_physics.csv")
    match_metrics = json.loads((args.output_dir / "v2_match_metrics.json").read_text(encoding="utf-8"))
    gan_locator_path = args.output_dir / "v2_gan_augmented_locator_smoke_metrics.json"
    gan_locator = json.loads(gan_locator_path.read_text(encoding="utf-8")) if gan_locator_path.exists() else {}
    s17 = pd.read_csv(args.s17_csv, usecols=["position_id", "s17_c_s", "s17_j_s"])
    payload = torch.load(args.residual_pth, map_location="cpu")

    summary = {
        "point_physics": {
            "rows": int(len(point)),
            "chirp_available_points": int(point["chirp_available"].sum()),
            "required_columns_present": all(
                column in point.columns
                for column in [
                    "distance_m",
                    "state",
                    "rho_chirp",
                    "rho_final",
                    "tau_d_us",
                    "mean_RSSI_phy",
                    "var_RSSI_phy",
                    "SNR_phy",
                ]
            ),
        },
        "s17": {
            "packets": int(len(s17)),
            "locations": int(s17["position_id"].nunique()),
            "c_s_mean": float(s17["s17_c_s"].mean()),
            "j_s_mean": float(s17["s17_j_s"].mean()),
        },
        "residual_gan_dataset": {
            key: list(value.shape)
            for key, value in payload.items()
            if hasattr(value, "shape")
        },
        "match_metrics": match_metrics,
        "gan_augmented_locator_smoke": gan_locator,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
