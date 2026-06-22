from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def check(condition: bool, name: str, evidence: object) -> dict:
    return {
        "name": name,
        "ok": bool(condition),
        "evidence": evidence,
    }


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that v2 implements the physics-guided fingerprint goal.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json-out", type=Path, default=Path("model/v2/output/v2_goal_validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    v2 = root / "model" / "v2"
    checks: list[dict] = []

    distance_path = root / "model" / "v1" / "output" / "location_distance_54points.csv"
    docs_distance_path = root / "docs" / "location_distance_54points.csv"
    theory_path = root / "docs" / "理论值生成流程.md"
    point_path = v2 / "output" / "v2_point_physics.csv"
    chirp_path = v2 / "output" / "v2_wideband_chirp_features.csv"
    s17_path = root / "data" / "processedData" / "usrp_preamble_fft_s17_54loc_20pkt_nonorm_relative_8sym.csv"
    residual_pth = v2 / "input" / "v2_residual_gan_dataset.pth"
    match_path = v2 / "output" / "v2_match_metrics.json"
    summary_path = v2 / "output" / "v2_experiment_summary.json"
    gan_smoke_path = v2 / "output" / "v2_gan_augmented_locator_smoke_metrics.json"

    v2_files = [
        v2 / "__init__.py",
        v2 / "v2_pipeline.py",
        v2 / "residual_gan.py",
        v2 / "gan_augmented_locator.py",
        v2 / "summarize_v2_results.py",
        v2 / "README.md",
    ]
    checks.append(check(v2.exists() and all(path.exists() for path in v2_files), "v2 source folder and scripts exist", [str(path) for path in v2_files]))

    distance_df = pd.read_csv(distance_path) if distance_path.exists() else pd.DataFrame()
    checks.append(check(len(distance_df) == 54 and "distance_m" in distance_df.columns, "uses 54-point location distance table", {"path": str(distance_path), "rows": len(distance_df), "columns": distance_df.columns.tolist()}))
    if docs_distance_path.exists() and distance_path.exists():
        checks.append(check(docs_distance_path.read_bytes() == distance_path.read_bytes(), "docs distance copy matches v1 distance table", {"docs": str(docs_distance_path), "v1": str(distance_path)}))
    else:
        checks.append(check(False, "docs distance copy matches v1 distance table", {"docs_exists": docs_distance_path.exists(), "v1_exists": distance_path.exists()}))
    checks.append(check(theory_path.exists() and "Step 1" in theory_path.read_text(encoding="utf-8"), "theory flow document exists", str(theory_path)))

    point_df = pd.read_csv(point_path) if point_path.exists() else pd.DataFrame()
    required_point_cols = [
        "distance_m",
        "state",
        "chirp_available",
        "rho_chirp",
        "tau_rms_chirp_us",
        "rho_final",
        "tau_d_us",
        "SNR_phy",
        "mean_RSSI_phy",
        "var_RSSI_phy",
    ]
    checks.append(check(len(point_df) == 54 and all(col in point_df.columns for col in required_point_cols), "point physics table covers 54 points and required physics columns", {"path": str(point_path), "rows": len(point_df), "required_columns": required_point_cols}))

    chirp_df = pd.read_csv(chirp_path) if chirp_path.exists() else pd.DataFrame()
    chirp_ok = (
        not chirp_df.empty
        and "path" in chirp_df.columns
        and chirp_df["path"].astype(str).str.contains("dong", case=False, regex=False).any()
        and {"rho_chirp", "tau_rms_chirp_us", "trusted_segments"}.issubset(chirp_df.columns)
    )
    checks.append(check(chirp_ok, "wideband chirp features are derived from dong/data_analysis", {"path": str(chirp_path), "rows": len(chirp_df), "columns": chirp_df.columns.tolist() if not chirp_df.empty else []}))

    s17_df = pd.read_csv(s17_path) if s17_path.exists() else pd.DataFrame()
    checks.append(check(len(s17_df) > 0 and {"s17_c_s", "s17_j_s"}.issubset(s17_df.columns), "USRP S17 features include C_S and J_S", {"path": str(s17_path), "rows": len(s17_df), "locations": int(s17_df["position_id"].nunique()) if "position_id" in s17_df.columns else 0}))

    residual_shapes = {}
    residual_ok = False
    if residual_pth.exists():
        payload = torch.load(residual_pth, map_location="cpu")
        residual_shapes = {key: list(value.shape) for key, value in payload.items() if hasattr(value, "shape")}
        residual_ok = all(key in payload for key in ["residual", "x_real", "x_phy", "condition", "label"]) and payload["residual"].shape[1] == 6
    checks.append(check(residual_ok, "residual GAN dataset contains residual/x_real/x_phy/condition/label", {"path": str(residual_pth), "shapes": residual_shapes}))

    match = read_json(match_path)
    fp = match.get("fingerprint_only", {})
    phy = match.get("physics_match", {})
    match_ok = bool(phy) and phy.get("accuracy", 0) >= fp.get("accuracy", 1) and phy.get("mean_distance_error_m", 1e9) <= fp.get("mean_distance_error_m", -1)
    checks.append(check(match_ok, "physics-aware match improves or matches fingerprint-only baseline", match))

    gan_smoke = read_json(gan_smoke_path)
    gan_ok = "real_plus_gan" in gan_smoke and gan_smoke["real_plus_gan"].get("accuracy", -1) >= gan_smoke.get("real_only", {}).get("accuracy", 1)
    checks.append(check(gan_ok, "GAN augmentation smoke test runs and improves or matches real-only locator", gan_smoke))

    v2_sources = "\n".join(path.read_text(encoding="utf-8") for path in v2_files if path.suffix == ".py")
    checks.append(check("DDPM" not in v2_sources and "ResidualGenerator" in v2_sources, "v2 uses GAN path and does not import DDPM", {"contains_residual_generator": "ResidualGenerator" in v2_sources, "contains_DDPM": "DDPM" in v2_sources}))

    summary = read_json(summary_path)
    checks.append(check(bool(summary), "experiment summary exists", {"path": str(summary_path), "keys": sorted(summary.keys()) if summary else []}))

    result = {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
