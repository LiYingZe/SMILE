import os
import pandas as pd

DATA_DIR = "./"

def remove_duplicates_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, dtype=str)
        
        df_deduplicated = df.drop_duplicates()
        
        if len(df) != len(df_deduplicated):
            df_deduplicated.to_csv(file_path, index=False)
            print(f"Deduplication complete: {file_path} ({len(df) - len(df_deduplicated)} rows removed)")
        else:
            print(f"No duplicates found: {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_all_csv_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith("phone"):
            file_path = os.path.join(directory, filename)
            remove_duplicates_from_csv(file_path)

process_all_csv_files(DATA_DIR)
