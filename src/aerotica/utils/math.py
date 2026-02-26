"""Mathematical utilities - بدون اعتماد على scipy."""

import numpy as np
from typing import Tuple
import math


def weibull_params(data: np.ndarray) -> Tuple[float, float]:
    """تقدير معاملات توزيع Weibull بطريقة مبسطة."""
    mean = np.mean(data)
    std = np.std(data)
    
    # تقدير تقريبي لمعامل الشكل k
    k = (std / mean) ** (-1.086)
    
    # معامل المقياس c
    # باستخدام تقريب لـ gamma function
    gamma_approx = 0.8862 + 0.112 * (k - 1.5)  # تقريب بسيط
    c = mean / gamma_approx
    
    return k, c


def wind_rose_stats(directions: np.ndarray,
                   speeds: np.ndarray = None,
                   n_bins: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """حساب إحصائيات وردة الرياح."""
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if speeds is not None:
        hist, _ = np.histogram(directions, bins=bin_edges, weights=speeds)
    else:
        hist, _ = np.histogram(directions, bins=bin_edges)
    
    if hist.sum() > 0:
        frequencies = hist / hist.sum()
    else:
        frequencies = hist
    
    return bin_centers, frequencies


def circular_mean(angles: np.ndarray) -> float:
    """حساب المتوسط الدائري للزوايا."""
    angles_rad = np.radians(angles)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    
    mean_angle = math.atan2(sin_sum, cos_sum)
    mean_angle = math.degrees(mean_angle) % 360
    
    return mean_angle
