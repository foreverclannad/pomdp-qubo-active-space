"""Check all extracted Supporting Information files against MANIFEST.json."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Missing or invalid file: " + name)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("File differs from the release: " + name)
    print(f"Verified {len(manifest['files'])} files for {manifest['release']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing MANIFEST.json")
    verify(parser.parse_args().directory)
