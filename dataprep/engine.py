import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from .logger import DataLogger
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import re

class DataPrepEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_rows = len(df)
        self.original_cols = len(df.columns)
        self.log = DataLogger()
        self._standardize_columns()

    def _standardize_columns(self):
        """Standardizes column names to snake_case and removes special characters."""
        tr_map = str.maketrans("çğışöüÇĞİŞÖÜ ", "cgisouCGISOU_")
        new_cols = []
        for col in self.df.columns:
            clean_col = str(col).translate(tr_map).lower()
            clean_col = re.sub(r'[^a-z0-9_]', '', clean_col)
            clean_col = re.sub(r'_+', '_', clean_col).strip('_')
            new_cols.append(clean_col)
        self.df.columns = new_cols
        self.log.action("Standardized column names to snake_case")

    def analyze(self):
        self.log.info(f"Dataset loaded: {self.original_rows} rows, {self.original_cols} columns")
        
        # Missing values
        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        cols_with_missing = missing[missing > 0]
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        
        # Health Score (basic)
        health_score = 100
        if self.original_rows > 0:
            health_score -= (total_missing / (self.original_rows * self.original_cols)) * 100
            health_score -= (duplicates / self.original_rows) * 100
        
        self.log.info(f"Data Health Score: {max(0, health_score):.1f}/100")
        
        if not cols_with_missing.empty:
            self.log.info(f"Missing values detected in {len(cols_with_missing)} columns")
        if duplicates > 0:
            self.log.info(f"Detected {duplicates} duplicate rows")

    def handle_outliers(self, threshold=3.0):
        """Clamps numerical outliers using Z-score."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std == 0 or pd.isna(std): continue
            
            lower = mean - threshold * std
            upper = mean + threshold * std
            
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if outliers > 0:
                self.df[col] = self.df[col].clip(lower, upper)
                self.log.action(f"Clamped {outliers} outliers in '{col}' (threshold={threshold})")

    def clean(self):
        # 1. Remove duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df.drop_duplicates(inplace=True)
            self.log.action(f"Removed {duplicates} duplicate rows")

        # 2. Handle outliers
        self.handle_outliers()

        # 3. Handle missing values
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    skew = self.df[col].skew()
                    strategy = "median" if abs(skew) > 1 else "mean"
                    fill_val = self.df[col].median() if strategy == "median" else self.df[col].mean()
                    self.df[col] = self.df[col].fillna(fill_val)
                    self.log.action(f"Filled {null_count} missing values in '{col}' using {strategy}")
                else:
                    fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else "unknown"
                    self.df[col] = self.df[col].fillna(fill_val)
                    self.log.action(f"Filled {null_count} missing values in '{col}' using mode ('{fill_val}')")

    def transform(self, goal: Optional[str] = None):
        # 4. Parse Dates & Extract Features
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    # Detect date-like columns
                    if any(x in col.lower() for x in ['date', 'time', 'year', 'month', 'tarih', 'gun', 'ay', 'yil']):
                        temp_dates = pd.to_datetime(self.df[col], errors='coerce')
                        if not temp_dates.isna().all():
                            self.df[col] = temp_dates
                            self.log.action(f"Transformed '{col}' to datetime")
                            
                            # Extract features if goal is ML
                            if goal in ["prediction", "classification"]:
                                self.df[f"{col}_year"] = self.df[col].dt.year
                                self.df[f"{col}_month"] = self.df[col].dt.month
                                self.df[f"{col}_day"] = self.df[col].dt.day
                                self.df[f"{col}_is_weekend"] = self.df[col].dt.dayofweek.isin([5, 6]).astype(int)
                                self.log.action(f"Extracted features from '{col}'")
                except:
                    pass

        # Only perform aggressive encoding/scaling if a specific ML goal is provided
        if goal in ["prediction", "classification"]:
            self.log.info(f"Applying {goal}-oriented transformations (ML Mode)")
            
            # 5. Encoding Categorical
            for col in self.df.select_dtypes(include=['object', 'category']).columns:
                nunique = self.df[col].nunique()
                if nunique < 10: # Low cardinality -> One-hot
                    self.df = pd.get_dummies(self.df, columns=[col], prefix=[col])
                    self.log.action(f"Encoded '{col}' using one-hot encoding")
                else: # Higher cardinality -> Label Encoding
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.log.action(f"Encoded '{col}' using label encoding")

            # 6. Scaling
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            # Exclude newly created year/month/day columns from scaling if wanted, 
            # but usually it's fine to scale them too.
            if not numeric_cols.empty:
                scaler = StandardScaler()
                self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
                self.log.action(f"Normalized {len(numeric_cols)} numeric columns")
        else:
            self.log.info("Skipping ML transformations for general cleaning.")

    def get_result(self) -> pd.DataFrame:
        return self.df
