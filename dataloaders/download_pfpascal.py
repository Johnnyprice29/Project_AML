import os
import zipfile
import requests
import argparse
from tqdm import tqdm

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}")
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def main():
    parser = argparse.ArgumentParser(description="Download PF-Pascal dataset")
    parser.add_argument("--root", type=str, default="./data", help="Root directory for data")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    pfpascal_path = os.path.join(args.root, "PF-Pascal")
    zip_path = os.path.join(args.root, "pfpascal.zip")
    
    # Official mirror
    url = "https://www.di.ens.fr/willow/research/proposal/dataset/PF-Pascal.zip"

    if os.path.exists(os.path.join(pfpascal_path, "PF-Pascal")):
        print(f"[INFO] PF-Pascal already found at {pfpascal_path}. Skipping.")
        return

    print(f"[INFO] Downloading PF-Pascal to {zip_path} ...")
    try:
        download_file(url, zip_path)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return

    print(f"[INFO] Extracting to {pfpascal_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(pfpascal_path)

    os.remove(zip_path)
    print(f"[INFO] Done. PF-Pascal ready.")

if __name__ == "__main__":
    main()
