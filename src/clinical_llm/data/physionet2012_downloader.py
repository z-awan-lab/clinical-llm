"""Download and prepare the PhysioNet/CinC Challenge 2012 dataset.

This dataset is publicly available (Open Data Commons Attribution Licence)
and does not require credentialing. It contains 12,000 ICU patient records
with vital signs and lab measurements over the first 48 hours of stay.
Sets A and B (8,000 patients) have public outcome labels; set C is
withheld by PhysioNet for benchmark evaluation only.

Usage:
    python -m clinical_llm.data.physionet2012_downloader \\
        --out-dir data/physionet2012

After downloading, run the loader:
    python -m clinical_llm.data.physionet2012_loader \\
        --raw-dir data/physionet2012/raw \\
        --out-dir data/physionet2012

Citation:
    Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital
    mortality of ICU patients: The PhysioNet/Computing in Cardiology
    Challenge 2012. Computing in Cardiology 2012; 39: 245-248.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

# Sets A and B both have publicly released outcomes. Set C does not — it was
# withheld for the original challenge's blind evaluation.
BASE_URL = "https://archive.physionet.org/pn3/challenge/2012"
FILES = {
    "set-a.tar.gz": f"{BASE_URL}/set-a.tar.gz",
    "set-b.tar.gz": f"{BASE_URL}/set-b.tar.gz",
    "Outcomes-a.txt": f"{BASE_URL}/Outcomes-a.txt",
    "Outcomes-b.txt": f"{BASE_URL}/Outcomes-b.txt",
}


def _progress_bar(block_num: int, block_size: int, total_size: int) -> None:
    """Simple textual progress reporter for urlretrieve."""
    downloaded = block_num * block_size
    pct = min(100, 100 * downloaded / max(total_size, 1))
    bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%")
    sys.stdout.flush()
    if pct >= 100:
        sys.stdout.write("\n")


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress reporting; skip if already present."""
    if dest.exists():
        print(f"  {dest.name}: already present, skipping.")
        return
    print(f"  {dest.name}: downloading...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress_bar)


def extract_tarball(tarball: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz to dest_dir, skipping if already extracted."""
    # Sets are extracted into directories named after the archive base name.
    extracted_marker = dest_dir / tarball.name.replace(".tar.gz", "")
    if extracted_marker.exists() and any(extracted_marker.iterdir()):
        print(f"  {tarball.name}: already extracted to {extracted_marker}/")
        return
    print(f"  {tarball.name}: extracting to {dest_dir}/")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/physionet2012"),
        help="Destination directory. Raw downloads go to <out-dir>/raw.",
    )
    args = parser.parse_args()

    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PhysioNet 2012 files to {raw_dir}/ ...")
    for name, url in FILES.items():
        download_file(url, raw_dir / name)

    print("\nExtracting record archives ...")
    for archive_name in ("set-a.tar.gz", "set-b.tar.gz"):
        extract_tarball(raw_dir / archive_name, raw_dir)

    print("\nDone.")
    print(f"Raw files in {raw_dir}/.")
    print(
        "Next step: python -m clinical_llm.data.physionet2012_loader "
        f"--raw-dir {raw_dir} --out-dir {args.out_dir}"
    )


if __name__ == "__main__":
    main()
