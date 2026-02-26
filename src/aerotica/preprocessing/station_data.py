"""Surface meteorological station data preprocessing."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from pathlib import Path
from datetime import datetime, timedelta


class StationDataProcessor:
    """Process surface meteorological station data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data = None
    
    def read_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Read station data from file.
        
        Supports CSV, NetCDF, and Excel formats.
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with station data
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.nc':
            import xarray as xr
            ds = xr.open_dataset(file_path)
            df = ds.to_dataframe().reset_index()
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.data = df
        return df
    
    def quality_control(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply quality control checks.
        
        - Remove outliers (> 3σ)
        - Fill missing values
        - Check consistency
        
        Args:
            df: Input DataFrame (uses self.data if None)
            
        Returns:
            Quality-controlled DataFrame
        """
        if df is None:
            df = self.data.copy()
        elif self.data is None:
            self.data = df.copy()
        
        # Remove outliers for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                df = df[abs(df[col] - mean) <= 3*std]
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining NaN with mean
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def resample(self, 
                df: pd.DataFrame,
                freq: str = '1H',
                timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Resample time series data.
        
        Args:
            df: Input DataFrame
            freq: Resampling frequency
            timestamp_col: Name of timestamp column
            
        Returns:
            Resampled DataFrame
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        
        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Set as index and resample
        df = df.set_index(timestamp_col)
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        other_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Resample numeric columns with mean
        df_numeric = df[numeric_cols].resample(freq).mean()
        
        # Resample other columns with first (for categorical)
        df_other = df[other_cols].resample(freq).first()
        
        # Combine
        df_resampled = pd.concat([df_numeric, df_other], axis=1)
        
        return df_resampled.reset_index()
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute basic statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q50': float(df[col].median()),
                'q75': float(df[col].quantile(0.75))
            }
        
        return stats
    
    def detect_spikes(self, 
                     df: pd.DataFrame,
                     column: str,
                     threshold: float = 4.0) -> pd.Series:
        """Detect spikes in time series.
        
        Args:
            df: Input DataFrame
            column: Column to check
            threshold: Spike detection threshold (in standard deviations)
            
        Returns:
            Boolean series with True for spikes
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        series = df[column]
        
        # Compute rolling statistics
        rolling_mean = series.rolling(window=10, center=True).mean()
        rolling_std = series.rolling(window=10, center=True).std()
        
        # Detect spikes
        spikes = abs(series - rolling_mean) > threshold * rolling_std
        
        return spikes.fillna(False)
    
    def merge_stations(self, 
                      dataframes: List[pd.DataFrame],
                      on: str = 'timestamp') -> pd.DataFrame:
        """Merge multiple station dataframes.
        
        Args:
            dataframes: List of DataFrames
            on: Column to merge on
            
        Returns:
            Merged DataFrame
        """
        from functools import reduce
        
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=on, how='outer'),
            dataframes
        )
        
        return merged
    
    def to_netcdf(self, df: pd.DataFrame, output_path: Path):
        """Save DataFrame to NetCDF.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
        """
        import xarray as xr
        
        # Convert to xarray
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        ds = xr.Dataset.from_dataframe(df)
        ds.to_netcdf(output_path)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"StationDataProcessor(config={self.config})"
