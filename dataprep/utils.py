import pandas as pd
import os
from .logger import DataLogger

from charset_normalizer import from_path

log = DataLogger()

def load_dataset(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            # Use charset-normalizer for pro encoding detection
            results = from_path(file_path).best()
            encoding = results.encoding if results else 'utf-8'
            coherence = results.percent_coherence if results else 0
            
            log.info(f"Detected encoding: {encoding} (coherence: {coherence:.2f}%)")
            return pd.read_csv(file_path, encoding=encoding)
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
            # use utf-8-sig to preserve Turkish characters for Excel
            df.to_csv(new_path, index=False, encoding='utf-8-sig')
        elif ext in ['.xls', '.xlsx']:
            df.to_excel(new_path, index=False)
        elif ext == '.json':
            df.to_json(new_path, orient='records', indent=4, force_ascii=False)
        
        log.success(f"Cleaned dataset saved to: {new_path}")
    except Exception as e:
        log.error(f"Failed to save dataset: {str(e)}")
        raise
