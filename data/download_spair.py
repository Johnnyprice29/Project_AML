"""
data/download_spair.py

Script to download and extract the SPair-71k dataset.

Usage:
    python data/download_spair.py --root ./datasets
"""

import os
import argparse
import tarfile
import urllib.request


SPAIR_URL = "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"


def download_spair71k(root: str):
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(root, "SPair-71k.tar.gz")
    extract_path = os.path.join(root, "SPair-71k")

    if os.path.exists(extract_path):
        print(f"[INFO] SPair-71k already found at {extract_path}. Skipping download.")
        return

    print(f"[INFO] Downloading SPair-71k to {tar_path} ...")
    urllib.request.urlretrieve(SPAIR_URL, tar_path, reporthook=_progress_hook)
    print("\n[INFO] Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(root)
    os.remove(tar_path)
    print(f"[INFO] Done. Dataset extracted to {extract_path}")


def _progress_hook(count, block_size, total_size):
    pct = min(count * block_size * 100 / total_size, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets",
                        help="Root directory where datasets will be stored")
    args = parser.parse_args()
    download_spair71k(args.root)
