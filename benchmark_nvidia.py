import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from algorithms import bubble_sort, quick_sort


def load_dataset(path: str) -> List[List[int]]:
    with open(path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of integer lists")
    
    for item in data:
        if not isinstance(item, list) or not all(isinstance(x, int) for x in item):
            raise ValueError("Each dataset entry must be a list of integers")
    
    return data


def benchmark(
    datasets: List[List[int]],
    algorithm: Callable[[List[int]], List[int]],
    name: str
) -> List[Tuple[int, int, float]]:
    results = []
    total = len(datasets)
    
    for idx, dataset in enumerate(datasets):
        original = dataset.copy()
        
        start = perf_counter()
        sorted_result = algorithm(original)
        elapsed = perf_counter() - start
        
        expected = sorted(dataset)
        assert sorted_result == expected, f"{name} failed on dataset {idx}"
        
        results.append((idx, len(dataset), elapsed))
        
        progress = (idx + 1) / total
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r  [{bar}] {idx + 1}/{total} ({progress * 100:.1f}%)', end='', flush=True)
    
    print()
    return results


def n_squared(n, a, b):
    return a * n**2 + b

def n_log_n(n, a, b):
    return a * n * np.log(n) + b


def fit_complexity(
    sizes: List[int],
    times: List[float],
    complexity_type: str
) -> Tuple[np.ndarray, Callable, str]:
    n = np.array(sizes)
    t = np.array(times)
    
    if complexity_type == "O(n²)":
        popt, _ = curve_fit(n_squared, n, t, maxfev=10000)
        fitted_func = lambda x: n_squared(x, *popt)
        formula = f"T(n) = {popt[0]:.2e}·n² + {popt[1]:.2e}"
    elif complexity_type == "O(n log n)":
        popt, _ = curve_fit(n_log_n, n, t, maxfev=10000)
        fitted_func = lambda x: n_log_n(x, *popt)
        formula = f"T(n) = {popt[0]:.2e}·n·log(n) + {popt[1]:.2e}"
    else:
        raise ValueError(f"Unknown complexity type: {complexity_type}")
    
    return popt, fitted_func, formula


def plot_comparison(
    all_results: dict[str, List[Tuple[int, int, float]]],
    complexity_map: dict[str, str],
    plots_dir: str
) -> Tuple[str, dict[str, str]]:
    os.makedirs(plots_dir, exist_ok=True)
    formulas = {}
    
    plt.figure(figsize=(12, 7))
    
    colors = {"Bubble Sort": "blue", "Quick Sort": "red"}
    
    for name, results in all_results.items():
        sizes = [r[1] for r in results]
        times = [r[2] for r in results]
        
        complexity_type = complexity_map.get(name, "O(n²)")
        _, fitted_func, formula = fit_complexity(sizes, times, complexity_type)
        formulas[name] = formula
        
        n_smooth = np.linspace(min(sizes), max(sizes), 200)
        t_smooth = fitted_func(n_smooth)
        
        color = colors.get(name, "black")
        plt.plot(sizes, times, marker="o", linestyle="", markersize=6, 
                label=f"{name} (Actual)", alpha=0.6, color=color)
        plt.plot(n_smooth, t_smooth, linestyle="-", linewidth=2, 
                label=f"{name} Regression: {complexity_type}", color=color)
    
    plt.xlabel("Dataset Size (n)", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Bubble Sort vs Quick Sort on NVIDIA Stock Data", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join(plots_dir, "nvidia_bubble_vs_quick.png")
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close()
    
    return filepath, formulas


def save_to_csv(
    all_results: dict[str, List[Tuple[int, int, float]]],
    csv_dir: str
) -> dict[str, str]:
    os.makedirs(csv_dir, exist_ok=True)
    csv_paths = {}
    
    for name, results in all_results.items():
        filename = f"{name.lower().replace(' ', '_')}_nvidia.csv"
        filepath = os.path.join(csv_dir, filename)
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset Index", "Size", "Time (s)"])
            for idx, size, elapsed in results:
                writer.writerow([idx, size, elapsed])
        
        csv_paths[name] = filepath
    
    return csv_paths


def write_comparison_report(
    all_results: dict[str, List[Tuple[int, int, float]]],
    plot_path: str,
    csv_paths: dict[str, str],
    formulas: dict[str, str],
    complexity_map: dict[str, str],
    report_path: str
) -> None:
    lines = ["# Bubble Sort vs Quick Sort on NVIDIA Stock Data\n"]
    lines.append("## Overview\n")
    lines.append("This report compares the performance of **Bubble Sort** (O(n²)) and **Quick Sort** (O(n log n)) ")
    lines.append("using real-world NVIDIA (NVDA) stock market data spanning multiple years. The datasets are derived ")
    lines.append("from historical closing prices and trading volumes, demonstrating sorting performance on realistic ")
    lines.append("financial data patterns.\n")
    
    lines.append("## Combined Performance Plot\n")
    rel_path = os.path.relpath(plot_path, os.path.dirname(report_path))
    lines.append(f"![Bubble Sort vs Quick Sort]({rel_path})\n")
    
    lines.append("## Algorithm Details\n")
    
    for name, results in all_results.items():
        lines.append(f"### {name}\n")
        
        complexity_type = complexity_map.get(name, "O(n²)")
        lines.append(f"**Theoretical Complexity:** {complexity_type}\n")
        
        if name in formulas:
            lines.append(f"**Regression Formula:**")
            lines.append(f"```")
            lines.append(formulas[name])
            lines.append(f"```\n")
        
        times = [r[2] for r in results]
        lines.append(f"**Statistics:**")
        lines.append(f"- Mean: {statistics.mean(times):.6f}s")
        lines.append(f"- Median: {statistics.median(times):.6f}s")
        lines.append(f"- Max: {max(times):.6f}s")
        lines.append(f"- Min: {min(times):.6f}s")
        
        if name in csv_paths:
            rel_csv_path = os.path.relpath(csv_paths[name], os.path.dirname(report_path))
            lines.append(f"\n**Data:** [{name} CSV]({rel_csv_path})\n")
    
    lines.append("## Analysis\n")
    
    bubble_times = [r[2] for r in all_results["Bubble Sort"]]
    quick_times = [r[2] for r in all_results["Quick Sort"]]
    
    lines.append(f"- **Speed Ratio (Max):** Bubble Sort is ~{max(bubble_times) / max(quick_times):.1f}x slower than Quick Sort at largest dataset")
    lines.append(f"- **Speed Ratio (Mean):** Bubble Sort is ~{statistics.mean(bubble_times) / statistics.mean(quick_times):.1f}x slower on average")
    
    lines.append("\n### Key Observations\n")
    lines.append("1. **Bubble Sort** exhibits clear O(n²) quadratic growth, becoming impractical as dataset size increases")
    lines.append("2. **Quick Sort** maintains efficient O(n log n) behavior, scaling well with larger financial datasets")
    lines.append("3. The performance gap widens dramatically with dataset size, confirming theoretical complexity predictions")
    lines.append("4. **Real-World Financial Data:** This benchmark uses NVIDIA stock data (prices and volumes), ")
    lines.append("   representing realistic sorting scenarios in financial applications like:")
    lines.append("   - Order book sorting in trading systems")
    lines.append("   - Time-series analysis requiring sorted price sequences")
    lines.append("   - Volume-weighted calculations on sorted data")
    lines.append("   - Market data aggregation and filtering\n")
    
    lines.append("### Data Source\n")
    lines.append("- **Ticker:** NVDA (NVIDIA Corporation)")
    lines.append("- **Period:** Maximum available history via Yahoo Finance")
    lines.append("- **Data Points:** Closing prices (scaled ×100) and trading volumes (scaled ÷1M)")
    lines.append("- **Use Case:** Demonstrates algorithm performance on real financial time-series data\n")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def ensure_output_dir(file_path: str) -> None:
    parent = Path(file_path).parent
    parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Compare Bubble Sort vs Quick Sort on NVIDIA data")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/nvidia_sets.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--csv-dir",
        default="data",
        help="Directory to save CSV files"
    )
    parser.add_argument(
        "--report",
        default="nvidia_bubble_vs_quick_report.md",
        help="Path to output report"
    )
    
    args = parser.parse_args()
    
    datasets = load_dataset(args.dataset)
    
    algorithms = [
        (bubble_sort, "Bubble Sort"),
        (quick_sort, "Quick Sort"),
    ]
    
    complexity_map = {
        "Bubble Sort": "O(n²)",
        "Quick Sort": "O(n log n)",
    }
    
    all_results = {}
    for algo, name in algorithms:
        print(f"Benchmarking {name}...")
        results = benchmark(datasets, algo, name)
        all_results[name] = results
    
    print(f"Generating comparison plot in {args.plots_dir}...")
    plot_path, formulas = plot_comparison(all_results, complexity_map, args.plots_dir)
    
    print(f"Saving results to CSV in {args.csv_dir}...")
    csv_paths = save_to_csv(all_results, args.csv_dir)
    
    print(f"Writing comparison report to {args.report}...")
    ensure_output_dir(args.report)
    write_comparison_report(all_results, plot_path, csv_paths, formulas, complexity_map, args.report)
    
    print("Comparison benchmark complete!")


if __name__ == "__main__":
    main()
