import requests
import os
from tqdm import tqdm

# Create directories
os.makedirs("/app/model_weights", exist_ok=True)

# File URLs and their destinations
files_to_download = {
    "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml": "/app/model_weights/config.yml",
    "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/Base-RCNN-FPN.yaml": "/app/Base-RCNN-FPN.yaml",
    "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl": "/app/model_weights/model_final.pkl"
}

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192 # 8 Kibibytes
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(destination)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Successfully downloaded {destination}")
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")
        exit(1)

# Download all files
for url, dest in files_to_download.items():
    download_file(url, dest)

print("All files downloaded successfully.")
