import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.config.load_config import load_config

def resize_image(
    input_csv_path,
    output_dir,
):
    input_csv_path = Path(input_csv_path)
    output_dir = Path(output_dir)
    
    config = load_config()
    target_size = config["data"]["image_size"] 
    
    print(f"[INFO] Starting process from: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Starting resizing process to size ({target_size}x{target_size})...")
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Resizing Images", unit="img"):
        
        original_path = Path(row['img_path'])
        label = row['label']
        
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        filename = original_path.name
        save_path = label_dir / filename
        
        try:
            img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
            
            if img is None:
                continue
                
            img_resized = cv2.resize(img, (target_size, target_size))
            
            cv2.imwrite(str(save_path), img_resized)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue
    
    print(f"[SUCCESS] Images successfully resized and saved at: {output_dir}")