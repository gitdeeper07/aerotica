"""Geospatial utilities - بدون اعتماد على scipy أو utm."""

import math
import numpy as np
from typing import Tuple


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
    """تحويل خط الطول والعرض إلى UTM تقريبي."""
    zone_number = int((lon + 180) / 6) + 1
    
    # تقدير تقريبي
    easting = (lon + 180) * 111319.9 * math.cos(math.radians(lat))
    northing = (lat + 90) * 111319.9
    zone_letter = 'N' if lat >= 0 else 'S'
    
    return easting, northing, zone_number, zone_letter


def haversine_distance(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> float:
    """حساب المسافة بين نقطتين على سطح الأرض."""
    R = 6371000  # نصف قطر الأرض بالمتر
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c
