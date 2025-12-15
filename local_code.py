### ---------- IMPORT LIBRARIES ----------
import os
import cv2
from tqdm import tqdm
import pandas as pd

base_dir = os.path.join(os.getcwd(), "model").replace(os.sep, "/")
print(f"Base directory: {base_dir}")
OUTPUT_DIR = os.path.join(base_dir, "assets", "chest_xray_resized").replace(os.sep, "/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resize_and_save_image(image_path, label, output_dir=OUTPUT_DIR, img_size=(224, 224)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read image {image_path}")
        
        img_resized = cv2.resize(img, img_size)
        
        label_dir = os.path.join("./assets", "chest_xray_resized", "NORMAL" if label == 0 else "PNEUMONIA").replace(os.sep, "/")
        os.makedirs(label_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        new_path = os.path.join(label_dir, filename).replace(os.sep, "/")
        
        cv2.imwrite(new_path, img_resized)
        
        new_size_kb = os.path.getsize(new_path) / 1024
        
        return {
            "resized_path": new_path,
            "label": label,
            "height": img_size[0],
            "width": img_size[1],
            "size_kb": new_size_kb
        }
        
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    

df_path = os.path.join(base_dir, "assets", "chest_xray.csv").replace(os.sep, "/")
df = pd.read_csv(df_path)

resized_data = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    result = resize_and_save_image(os.path.join(base_dir, row['path']).replace(os.sep, "/"), row['label'])
    
    if result:
        resized_data.append(result)
        
resized_df = pd.DataFrame(resized_data)
resized_df.to_csv(os.path.join(base_dir, "assets", "chest_xray_resized.csv"), index=False)
