#!/usr/bin/env python3
"""Run ACO v4 with chirp fields disabled for a LoRa-only template ablation."""

from __future__ import annotations

import json

import run_aco_v4_on_split as runner


def prepare_lora_only_fields(_args, labels):
    return (
        {},
        {},
        {
            "ablation": "lora_only_no_chirp_fields",
            "labels_requiring_template": len(set(labels)),
            "template_sources": {"lora_only_empirical_global_fallback": len(set(labels))},
            "structure_sources": {"lora_only_structure_default": len(set(labels))},
        },
    )


def main() -> None:
    runner.aco4.aco2.prepare_chirp_fields = prepare_lora_only_fields
    metadata = runner.run(runner.parse_args())
    print(json.dumps(metadata["sample_counts"], indent=2, ensure_ascii=False))
    for row in metadata["summary"]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {metadata['args']['output_dir']}")


if __name__ == "__main__":
    main()
