import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from src.config.load_config import load_config 

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def build_image_to_csv(
    data_dir,
    output_dir,
    namespace="dataset"
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    labels = config["data"]["labels"]

    records = []

    print(f"[INFO] Starting process from: {data_dir}")

    for label in labels:
        label_dir = data_dir / label

        if not label_dir.exists():
            print(f"[WARNING] Label folder not found: {label_dir}")
            continue

        image_files = list(label_dir.iterdir())

        for img_path in tqdm(image_files, desc=f"Processing {label}", unit="img"):
            
            if not img_path.is_file():
                continue
            
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                img_size_kb = img_path.stat().st_size / 1024

                records.append({
                    "img_path": img_path.as_posix(),
                    "height": height,
                    "width": width,
                    "image_size_kb": round(img_size_kb, 3),
                    "label": label
                })

            except Exception as e:
               print(f"[ERROR] Failed to process {img_path.name}: {e}")

    df = pd.DataFrame(records)

    if df.empty:
        print("[WARNING] No images found! Please check your folder path.")
        return None

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_csv = output_dir / f"{namespace}.csv"
    df.to_csv(output_csv, index=False)

    print(f"[SUCCESS] CSV successfully saved at: {output_csv}")
    print(f"[INFO] Total Images: {len(df)}")
    print(f"[INFO] Class Distribution:\n{df['label'].value_counts()}")