"""Download TinyStories dataset for FluidLM training."""
import os
import requests
from tqdm import tqdm

def download_tinystories():
    # Raw mirror URL for TinyStories (.txt or .jsonl format)
    # Starting with a sample (~30MB)
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, "tinystories.txt")

    if os.path.exists(target_path):
        print(f"OK: {target_path} already exists.")
        return

    print(f"Downloading TinyStories to {target_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(target_path, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc="Download"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print("OK: Done. FluidLM can now train on stories.")

if __name__ == "__main__":
    download_tinystories()