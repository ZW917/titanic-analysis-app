"""Read-only verification of the user's protected original project files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = ROOT / ".local" / "source_integrity.json"
    if not manifest.exists():
        raise SystemExit("No local source manifest. This check is only for the original build computer.")
    rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
    checks = []
    for row in rows:
        path = Path(row["path"])
        exists = path.is_file()
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest().upper() if exists else None
        checks.append({"path": str(path), "unchanged": current_hash == row["sha256"], "exists": exists})
    summary = {"checked": len(checks), "all_unchanged": all(x["unchanged"] for x in checks), "files": checks}
    (ROOT / ".local" / "source_integrity_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"checked": summary["checked"], "all_unchanged": summary["all_unchanged"]}))
    if not summary["all_unchanged"]:
        raise SystemExit("A protected original differs from its recorded starting hash.")


if __name__ == "__main__":
    main()
