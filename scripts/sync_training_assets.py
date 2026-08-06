"""Add fully audited candidate WAV files to the training whitelist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIT_PATH = DATA_DIR / "audio_asset_audit.json"
ASSETS_PATH = DATA_DIR / "training_assets.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvids", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    by_bvid = {str(record["bvid"]): record for record in audit}
    requested = list(dict.fromkeys(args.bvids))
    missing = [bvid for bvid in requested if bvid not in by_bvid]
    if missing:
        parser.error("BVIDs are absent from the audit: " + ", ".join(missing))

    rejected = [
        f"{bvid}={by_bvid[bvid].get('status')}"
        for bvid in requested
        if by_bvid[bvid].get("status") != "ready"
    ]
    if rejected:
        parser.error("assets are not ready: " + ", ".join(rejected))

    current = [
        line.strip()
        for line in ASSETS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    names = list(current)
    for bvid in requested:
        wav_path = Path(str(by_bvid[bvid]["wav_path"]))
        if not wav_path.is_file():
            parser.error(f"audited WAV no longer exists: {wav_path}")
        if wav_path.name not in names:
            names.append(wav_path.name)

    additions = names[len(current) :]
    if not additions:
        print("training whitelist already contains all requested assets")
        return
    for name in additions:
        print(f"add: {name}")
    if args.dry_run:
        return

    tmp_path = ASSETS_PATH.with_name(f"{ASSETS_PATH.name}.tmp")
    tmp_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    tmp_path.replace(ASSETS_PATH)
    print(f"training assets: {len(current)} -> {len(names)}")


if __name__ == "__main__":
    main()
