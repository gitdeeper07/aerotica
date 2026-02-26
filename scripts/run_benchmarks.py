#!/usr/bin/env python3
"""Run performance benchmarks."""

import time
import json
import numpy as np
from pathlib import Path
import argparse

from aerotica.ake import AKEComposite
from aerotica.parameters import KED, TII, VSR, THD


def benchmark_parameter_computation(n_runs: int = 100):
    """Benchmark parameter computation speed."""
    times = []
    
    ked = KED()
    wind_data = np.random.randn(1000) * 2 + 10
    
    for _ in range(n_runs):
        start = time.perf_counter()
        ked.compute(wind_data)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }


def benchmark_ake_computation(n_runs: int = 100):
    """Benchmark AKE composite computation."""
    times = []
    
    ake = AKEComposite("benchmark", "temperate", "test")
    params = {
        "KED": 0.8, "TII": 0.7, "VSR": 0.8,
        "AOD": 0.3, "THD": 0.7, "PGF": 0.6,
        "HCI": 0.6, "ASI": 0.7, "LRC": 0.4
    }
    ake.load_parameters(params)
    
    for _ in range(n_runs):
        start = time.perf_counter()
        ake.compute()
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }


def benchmark_gust_detection(n_runs: int = 100):
    """Benchmark gust detection speed."""
    from aerotica.alerts import GustPreAlertEngine
    
    times = []
    engine = GustPreAlertEngine()
    
    # Mock observations
    import pandas as pd
    obs = pd.DataFrame({
        'wind_speed': np.random.randn(100) * 2 + 10,
        'wind_direction': np.random.rand(100) * 360,
        'temperature': np.random.randn(100) * 2 + 20,
    })
    
    for _ in range(n_runs):
        start = time.perf_counter()
        engine.evaluate(obs)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }


def main():
    """Run all benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default='benchmarks/results.json')
    parser.add_argument('--runs', type=int, default=100)
    args = parser.parse_args()
    
    print("🚀 Running AEROTICA benchmarks...")
    print(f"   Runs per benchmark: {args.runs}")
    
    results = {
        'timestamp': time.time(),
        'runs': args.runs,
        'benchmarks': {}
    }
    
    # Parameter computation
    print("\n📊 Benchmarking parameter computation...")
    results['benchmarks']['parameter'] = benchmark_parameter_computation(args.runs)
    print(f"   Mean: {results['benchmarks']['parameter']['mean_ms']:.2f} ms")
    
    # AKE computation
    print("\n📊 Benchmarking AKE computation...")
    results['benchmarks']['ake'] = benchmark_ake_computation(args.runs)
    print(f"   Mean: {results['benchmarks']['ake']['mean_ms']:.2f} ms")
    
    # Gust detection
    print("\n📊 Benchmarking gust detection...")
    results['benchmarks']['gust'] = benchmark_gust_detection(args.runs)
    print(f"   Mean: {results['benchmarks']['gust']['mean_ms']:.2f} ms")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
