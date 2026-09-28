"""Create an upload-ready ZIP from an explicit list of public project files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = [
    "app.py", "ui.py", "pipeline.py", "README.md", "requirements.txt",
    "requirements-dev.txt", "pytest.ini", ".gitignore", ".gitattributes", "start_app.cmd",
    ".streamlit/config.toml",
]
FOLDERS = {"data": {".csv", ".md"}, "artifacts": {".csv", ".json", ".joblib"},
           "docs": {".md"}, "scripts": {".py"}, "tests": {".py"}}


def main():
    files = [ROOT / name for name in TOP_LEVEL]
    for folder, extensions in FOLDERS.items():
        files.extend(path for path in (ROOT / folder).iterdir()
                     if path.is_file() and path.suffix in extensions)
    files = sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    target = output / "titanic_streamlit_public.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)
    manifest = {"archive": target.name, "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "files": [path.relative_to(ROOT).as_posix() for path in files]}
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archive": str(target), "files": len(files), "bytes": manifest["bytes"]}))


if __name__ == "__main__":
    main()
