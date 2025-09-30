import os
import shutil
import pandas as pd

# --- Configuration ---
csv_path = "/Users/saikrishna/Desktop/MLLM/nudge-x/research/data/metadata/Open_Mines_Coordinates_v12.csv"
source_folder = "/Users/saikrishna/Desktop/MLLM/nudge-x/research/data/sites"
output_folder = "/Users/saikrishna/Desktop/MLLM/nudge-x/research/data/images"       # Output folder for matched PNGs

# --- 1. Read CSV and filter ---
df = pd.read_csv(csv_path)

# Filter rows where 'metadata' is NOT empty
df_filtered = df[df['metadata'].notna() & (df['metadata'].astype(str).str.strip() != "")]

# --- 2. Get valid names (lowercased for case-insensitive matching) ---
valid_names = df_filtered['Mine name'].astype(str).str.lower().unique()

# --- 3. Create output folder ---
os.makedirs(output_folder, exist_ok=True)

# --- 4. Match and copy PNG files ---
count = 0
for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(".png"):
        lower_file = file_name.lower()
        for name in valid_names:
            if name in lower_file:
                src = os.path.join(source_folder, file_name)
                dst = os.path.join(output_folder, file_name)
                shutil.copy2(src, dst)
                count += 1
                break  # prevent duplicate copies if multiple names match

print(f"✅ Done! Copied {count} PNG files to '{output_folder}'")
