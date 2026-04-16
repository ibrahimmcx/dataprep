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

    def analyze(self):
        self.log.info(f"Dataset loaded: {self.original_rows} rows, {self.original_cols} columns")
        
        # Missing values
        missing = self.df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        if not cols_with_missing.empty:
            self.log.info(f"Missing values detected in {len(cols_with_missing)} columns")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.log.info(f"Detected {duplicates} duplicate rows")

    def clean(self):
        # 1. Remove duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df.drop_duplicates(inplace=True)
            self.log.action(f"Removed {duplicates} duplicate rows")

        # 2. Handle missing values
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # Check skewness to decide between mean and median
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
        # 3. Parse Dates
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    # Attempt to parse as date if it looks like one
                    if any(x in col.lower() for x in ['date', 'time', 'year', 'month']):
                        self.df[col] = pd.to_datetime(self.df[col])
                        self.log.action(f"Transformed '{col}' to datetime")
                except:
                    pass

        # 4. Encoding Categorical
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            nunique = self.df[col].nunique()
            if nunique < 10: # Low cardinality -> One-hot
                self.df = pd.get_dummies(self.df, columns=[col], prefix=[col])
                self.log.action(f"Encoded '{col}' using one-hot encoding")
            else: # Higher cardinality -> Label Encoding
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.log.action(f"Encoded '{col}' using label encoding")

        # 5. Goal oriented scaling
        if goal in ["prediction", "classification"]:
            self.log.info(f"Applying {goal}-oriented transformations")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            self.log.action(f"Normalized numeric columns using StandardScaler")

    def get_result(self) -> pd.DataFrame:
        return self.df
