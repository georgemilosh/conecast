#!/usr/bin/env python
"""Download the WSA+ model checkpoint (`wsaplus.pt`) from Zenodo.

The WSA+ speed-map model needs a trained PyTorch checkpoint (~317 MB) that is too large
to ship in git. It is published on Zenodo (DOI 10.5281/zenodo.16883042). This script
downloads it to `data_dir/sw/wsaplus.pt`, the location `generate_huxt_input.py` and the
notebooks expect.

    python scripts/fetch_wsaplus_checkpoint.py            # -> data_dir/sw/wsaplus.pt
    python scripts/fetch_wsaplus_checkpoint.py --force    # re-download
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Direct file-content endpoint for the Zenodo record (DOI 10.5281/zenodo.16883042).
ZENODO_URL = "https://zenodo.org/api/records/16883042/files/wsaplus.pt/content"
EXPECTED_MIN_BYTES = 100_000_000  # ~317 MB; guard against downloading an HTML error page

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DEST = BASE_DIR / "data_dir" / "sw" / "wsaplus.pt"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    done = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * done / total_size)
        sys.stdout.write(f"\r  {done / 1e6:7.0f} / {total_size / 1e6:.0f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {done / 1e6:7.0f} MB")
    sys.stdout.flush()


def download(dest: Path = DEFAULT_DEST, url: str = ZENODO_URL, force: bool = False) -> Path:
    """Download the checkpoint to `dest` (skips if it already exists, unless `force`)."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"{dest} already exists ({dest.stat().st_size / 1e6:.0f} MB); pass force=True to re-download.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading WSA+ checkpoint (~317 MB) from Zenodo:\n  {url}\n  -> {dest}")
    urllib.request.urlretrieve(url, tmp, _progress)
    print()

    size = tmp.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is only {size / 1e6:.1f} MB - expected ~317 MB. "
            f"The download likely failed; check the URL or your connection."
        )
    tmp.replace(dest)
    print(f"Done: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Output path (default: data_dir/sw/wsaplus.pt).")
    parser.add_argument("--url", default=ZENODO_URL, help="Override the download URL.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args()
    download(dest=args.dest, url=args.url, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
