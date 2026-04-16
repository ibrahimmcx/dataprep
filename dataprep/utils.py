import pandas as pd
import os
from .logger import DataLogger

log = DataLogger()

def load_dataset(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        log.error(f"Failed to load dataset: {str(e)}")
        raise

def save_dataset(df: pd.DataFrame, original_path: str):
    base, ext = os.path.splitext(original_path)
    new_path = f"{base}_cleaned{ext}"
    try:
        if ext == '.csv':
            df.to_csv(new_path, index=False)
        elif ext in ['.xls', '.xlsx']:
            df.to_excel(new_path, index=False)
        elif ext == '.json':
            df.to_json(new_path, orient='records', indent=4)
        
        log.success(f"Cleaned dataset saved to: {new_path}")
    except Exception as e:
        log.error(f"Failed to save dataset: {str(e)}")
        raise
