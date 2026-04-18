import os
import zipfile
import requests
import argparse
from tqdm import tqdm

def download_file(url, destination):
    # Aggiungiamo un User-Agent per evitare blocchi dal server
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
    
    if response.status_code != 200:
        raise Exception(f"Server returned status code {response.status_code}")
        
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading PF-Pascal")
    
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
    
    # Official URL
    url = "https://www.di.ens.fr/willow/research/proposal/dataset/PF-Pascal.zip"

    if os.path.exists(os.path.join(pfpascal_path, "PF-Pascal")):
        print(f"[INFO] PF-Pascal already found at {pfpascal_path}. Skipping.")
        return

    print(f"[INFO] Downloading PF-Pascal from official mirror...")
    try:
        download_file(url, zip_path)
    except Exception as e:
        print(f"[WARNING] Official link failed: {e}. Trying alternative mirror...")
        # Mirror alternativo (es. se Willow è giù)
        url_alt = "http://cs.unc.edu/~jlt/data/PF-Pascal.zip" # Esempio di mirror se disponibile
        try:
            download_file(url_alt, zip_path)
        except Exception as e2:
            print(f"[ERROR] All download attempts failed.")
            return

    print(f"[INFO] Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pfpascal_path)
        print(f"[INFO] Extraction successful.")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}. The downloaded file might be corrupted.")
        if os.path.exists(zip_path):
            with open(zip_path, 'r', errors='ignore') as f:
                print(f"File content (first 100 chars): {f.read(100)}")
        return
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    print(f"[INFO] Done. PF-Pascal ready.")

if __name__ == "__main__":
    main()
