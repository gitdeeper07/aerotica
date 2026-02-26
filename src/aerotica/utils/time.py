"""Time utilities."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Optional, Tuple


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def timestamp_to_datetime(ts: float) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(ts)


def get_season(date: Union[datetime, pd.Timestamp]) -> str:
    """Get season for a given date.
    
    Args:
        date: Date
        
    Returns:
        Season: 'spring', 'summer', 'fall', 'winter'
    """
    month = date.month
    
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'fall'
    else:
        return 'winter'


def time_of_day(dt: datetime) -> str:
    """Get time of day category.
    
    Args:
        dt: Datetime
        
    Returns:
        'dawn', 'morning', 'noon', 'afternoon', 'dusk', 'night'
    """
    hour = dt.hour + dt.minute / 60
    
    if 5 <= hour < 7:
        return 'dawn'
    elif 7 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 13:
        return 'noon'
    elif 13 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 19:
        return 'dusk'
    else:
        return 'night'


def is_daylight(dt: datetime, sunrise: Optional[datetime] = None,
                sunset: Optional[datetime] = None) -> bool:
    """Check if it's daylight.
    
    Args:
        dt: Datetime
        sunrise: Sunrise time (if None, uses approximate)
        sunset: Sunset time (if None, uses approximate)
        
    Returns:
        True if daylight
    """
    if sunrise is None or sunset is None:
        # Approximate: 6am to 6pm
        return 6 <= dt.hour < 18
    
    return sunrise <= dt <= sunset


def date_range(start: datetime, end: datetime, 
               freq: str = '1h') -> pd.DatetimeIndex:
    """Generate date range."""
    return pd.date_range(start, end, freq=freq)


def resample_time_series(df: pd.DataFrame, 
                        freq: str = '1h',
                        agg_func: str = 'mean') -> pd.DataFrame:
    """Resample time series data.
    
    Args:
        df: DataFrame with datetime index
        freq: Resampling frequency
        agg_func: Aggregation function
        
    Returns:
        Resampled DataFrame
    """
    return df.resample(freq).agg(agg_func)


def running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Compute running mean.
    
    Args:
        x: Input array
        window: Window size
        
    Returns:
        Running mean array
    """
    return np.convolve(x, np.ones(window)/window, mode='same')


def detect_outliers(series: pd.Series, 
                   threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method.
    
    Args:
        series: Input series
        threshold: Z-score threshold
        
    Returns:
        Boolean series with True for outliers
    """
    zscore = (series - series.mean()) / series.std()
    return abs(zscore) > threshold
