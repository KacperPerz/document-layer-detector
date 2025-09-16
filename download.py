import requests
import os
from tqdm import tqdm

# Create directories
os.makedirs("/app/model_weights", exist_ok=True)

# PubLayNet Faster R-CNN R50 FPN 3x weights used by layoutparser
# Use the official Dropbox link (matches iopath cache key seen in logs)
PUBL_WEIGHT_URL = "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1"
DEST = "/app/model_weights/publaynet_frcnn_r50_fpn_3x.pth"


def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192
            tmp_dest = destination + ".part"
            with open(tmp_dest, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            os.replace(tmp_dest, destination)
        print(f"Successfully downloaded {destination}")
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")
        # Don't hard fail; allow runtime fallback in case this mirror is down
        # exit(1)


if not os.path.exists(DEST):
    download_file(PUBL_WEIGHT_URL, DEST)
else:
    print(f"Weights already present at {DEST}")

print("Download step finished.")
