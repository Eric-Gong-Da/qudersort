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

from algorithms import bubble_sort, quick_sort, bucket_sort


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

def n_plus_k(n, a, b):
    return a * n + b


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
    elif complexity_type == "O(n+k)":
        popt, _ = curve_fit(n_plus_k, n, t, maxfev=10000)
        fitted_func = lambda x: n_plus_k(x, *popt)
        formula = f"T(n) = {popt[0]:.2e}·n + {popt[1]:.2e}"
    else:
        raise ValueError(f"Unknown complexity type: {complexity_type}")
    
    return popt, fitted_func, formula


def plot_results(
    all_results: dict[str, List[Tuple[int, int, float]]],
    complexity_map: dict[str, str],
    plots_dir: str
) -> Tuple[dict[str, str], dict[str, str]]:
    os.makedirs(plots_dir, exist_ok=True)
    plot_paths = {}
    formulas = {}
    
    for name, results in all_results.items():
        sizes = [r[1] for r in results]
        times = [r[2] for r in results]
        
        complexity_type = complexity_map.get(name, "O(n²)")
        _, fitted_func, formula = fit_complexity(sizes, times, complexity_type)
        formulas[name] = formula
        
        n_smooth = np.linspace(min(sizes), max(sizes), 200)
        t_smooth = fitted_func(n_smooth)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, marker="o", linestyle="", markersize=8, label="Actual Data", alpha=0.7)
        plt.plot(n_smooth, t_smooth, linestyle="-", linewidth=2, label=f"Regression: {complexity_type}", color="red")
        plt.xlabel("Dataset Size", fontsize=12)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.title(f"{name} Performance", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()
        
        plot_paths[name] = filepath
    
    return plot_paths, formulas


def save_to_csv(
    all_results: dict[str, List[Tuple[int, int, float]]],
    csv_dir: str
) -> dict[str, str]:
    os.makedirs(csv_dir, exist_ok=True)
    csv_paths = {}
    
    for name, results in all_results.items():
        filename = f"{name.lower().replace(' ', '_')}.csv"
        filepath = os.path.join(csv_dir, filename)
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset Index", "Size", "Time (s)"])
            for idx, size, elapsed in results:
                writer.writerow([idx, size, elapsed])
        
        csv_paths[name] = filepath
    
    return csv_paths


def write_report(
    all_results: dict[str, List[Tuple[int, int, float]]],
    plot_paths: dict[str, str],
    csv_paths: dict[str, str],
    formulas: dict[str, str],
    complexity_map: dict[str, str],
    report_path: str
) -> None:
    lines = ["# Sorting Algorithm Benchmark Report\n"]
    
    for name, results in all_results.items():
        lines.append(f"\n## {name}\n")
        
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
        
        if name in csv_paths:
            rel_path = os.path.relpath(csv_paths[name], os.path.dirname(report_path))
            lines.append(f"\n**Data:** [{name} CSV]({rel_path})")
        
        if name in plot_paths:
            rel_path = os.path.relpath(plot_paths[name], os.path.dirname(report_path))
            lines.append(f"\n![{name} Performance]({rel_path})")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def ensure_output_dir(file_path: str) -> None:
    parent = Path(file_path).parent
    parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark sorting algorithms")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/sample_sets.json",
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
        default="report.md",
        help="Path to output report"
    )
    
    args = parser.parse_args()
    
    datasets = load_dataset(args.dataset)
    
    algorithms = [
        (bubble_sort, "Bubble Sort"),
        (quick_sort, "Quick Sort"),
        (bucket_sort, "Bucket Sort"),
    ]
    
    complexity_map = {
        "Bubble Sort": "O(n²)",
        "Quick Sort": "O(n log n)",
        "Bucket Sort": "O(n+k)",
    }
    
    all_results = {}
    for algo, name in algorithms:
        print(f"Benchmarking {name}...")
        results = benchmark(datasets, algo, name)
        all_results[name] = results
    
    print(f"Generating plots in {args.plots_dir}...")
    plot_paths, formulas = plot_results(all_results, complexity_map, args.plots_dir)
    
    print(f"Saving results to CSV in {args.csv_dir}...")
    csv_paths = save_to_csv(all_results, args.csv_dir)
    
    print(f"Writing report to {args.report}...")
    ensure_output_dir(args.report)
    write_report(all_results, plot_paths, csv_paths, formulas, complexity_map, args.report)
    
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
