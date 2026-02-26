"""Input/Output utilities."""

import json
import yaml
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import xarray as xr
import rasterio


def read_netcdf(file_path: Union[str, Path]) -> xr.Dataset:
    """Read NetCDF file.
    
    Args:
        file_path: Path to NetCDF file
        
    Returns:
        xarray Dataset
    """
    return xr.open_dataset(file_path)


def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read CSV file.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pandas DataFrame
    """
    return pd.read_csv(file_path, **kwargs)


def read_geotiff(file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """Read GeoTIFF file.
    
    Args:
        file_path: Path to GeoTIFF file
        
    Returns:
        (data, metadata)
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'resolution': src.res
        }
    return data, metadata


def read_json(file_path: Union[str, Path]) -> Dict:
    """Read JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def read_yaml(file_path: Union[str, Path]) -> Dict:
    """Read YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def write_json(data: Any, file_path: Union[str, Path], indent: int = 2):
    """Write JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def write_yaml(data: Any, file_path: Union[str, Path]):
    """Write YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def write_csv(data: pd.DataFrame, file_path: Union[str, Path], **kwargs):
    """Write CSV file."""
    data.to_csv(file_path, index=False, **kwargs)


def write_netcdf(data: xr.Dataset, file_path: Union[str, Path]):
    """Write NetCDF file."""
    data.to_netcdf(file_path)


def ensure_dir(directory: Union[str, Path]):
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def list_files(directory: Union[str, Path], 
              pattern: str = "*") -> List[Path]:
    """List files in directory matching pattern."""
    return list(Path(directory).glob(pattern))


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def read_lines(file_path: Union[str, Path]) -> List[str]:
    """Read lines from text file."""
    with open(file_path, 'r') as f:
        return f.readlines()


def write_lines(lines: List[str], file_path: Union[str, Path]):
    """Write lines to text file."""
    with open(file_path, 'w') as f:
        f.writelines(lines)


def parse_filename(file_path: Union[str, Path]) -> Dict[str, str]:
    """Parse filename into components.
    
    Returns:
        {'name':, 'stem':, 'suffix':, 'parent':}
    """
    path = Path(file_path)
    return {
        'name': path.name,
        'stem': path.stem,
        'suffix': path.suffix,
        'parent': str(path.parent)
    }


def validate_file(file_path: Union[str, Path], 
                 check_size: bool = True) -> bool:
    """Validate file exists and has content."""
    path = Path(file_path)
    if not path.exists():
        return False
    if check_size and path.stat().st_size == 0:
        return False
    return True
