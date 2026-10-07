"""Validate the machine-readable frontier evidence-track contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TRACKS = {
    "rsi_research_automation",
    "long_horizon_rl",
    "coding_rl",
    "rl_infrastructure",
    "end_to_end_model_building",
    "post_training_rl",
}


def validate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != TRACKS:
        raise ValueError("exact six frontier tracks required")
    order = payload.get("priority_order")
    if not isinstance(order, list) or set(order) != TRACKS or len(order) != len(TRACKS):
        raise ValueError("priority_order must contain each track exactly once")
    for name, track in tracks.items():
        if not isinstance(track.get("target"), str) or not track["target"].strip():
            raise ValueError(f"{name}: target required")
        existing = track.get("existing")
        required = track.get("required_evidence")
        if not isinstance(existing, list) or not existing:
            raise ValueError(f"{name}: existing evidence list required")
        if not isinstance(required, list) or not required:
            raise ValueError(f"{name}: required_evidence list required")
        status = track.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"{name}: status required")
        # Fail closed against accidental evidence-ready claims in the roadmap.
        if status in {"ready", "evidence_ready", "complete"}:
            raise ValueError(
                f"{name}: roadmap cannot declare readiness; readiness requires raw evidence"
            )
    return {
        "valid": True,
        "tracks": len(tracks),
        "priority_order": order,
        "claim_policy": payload["claim_policy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("configs/frontier_tracks_v1.json"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
