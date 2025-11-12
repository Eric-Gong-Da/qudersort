"""
===========================================================
Sorting Algorithms Performance Analysis Tool
===========================================================

Author: Gong Da
Student ID: 11536511
Last Updated: 2025-11-12
Purpose: This program benchmarks and analyzes the performance of three sorting algorithms:
         Bubble Sort (O(n²)), Quick Sort (O(n log n)), and Bucket Sort (O(n+k)).
         It evaluates these algorithms on both synthetic random datasets and real-world 
         NVIDIA (NVDA) stock market data to provide empirical guidance for algorithm selection.
===========================================================
"""

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
    """
    Benchmarks a sorting algorithm on a collection of datasets.
    
    Args:
        datasets (List[List[int]]): A list of datasets to sort
        algorithm (Callable[[List[int]], List[int]]): The sorting algorithm to benchmark
        name (str): Name of the algorithm for reporting purposes
        
    Returns:
        List[Tuple[int, int, float]]: List of (dataset_index, dataset_size, execution_time) tuples
        
    Algorithm:
        1. For each dataset in the collection:
           a. Create a copy to avoid modifying the original
           b. Measure execution time of the sorting algorithm
           c. Verify correctness by comparing with Python's built-in sort
           d. Record results and show progress
    """
    # Initialize list to store benchmark results
    # Each entry will be (dataset_index, dataset_size, execution_time)
    results = []
    
    # Get total number of datasets for progress tracking
    total = len(datasets)
    
    # Iterate through each dataset to benchmark the algorithm
    for idx, dataset in enumerate(datasets):
        # Create a copy of the dataset to avoid modifying the original
        # This ensures each algorithm runs on the same input data
        original = dataset.copy()
        
        # Record start time for execution time measurement
        # perf_counter provides the highest available resolution timing
        start = perf_counter()
        
        # Execute the sorting algorithm on the dataset copy
        sorted_result = algorithm(original)
        
        # Calculate execution time by taking difference from start time
        elapsed = perf_counter() - start
        
        # Verify correctness by comparing with Python's built-in sorted function
        # This ensures our implementations are working correctly
        expected = sorted(dataset)
        assert sorted_result == expected, f"{name} failed on dataset {idx}"
        
        # Store the benchmark results as (dataset_index, dataset_size, execution_time)
        results.append((idx, len(dataset), elapsed))
        
        # Display progress bar to show benchmark status
        # This is helpful for long-running benchmarks
        progress = (idx + 1) / total
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r  [{bar}] {idx + 1}/{total} ({progress * 100:.1f}%)', end='', flush=True)
    
    # Print newline to move cursor to next line after progress bar
    print()
    
    # Return the collected benchmark results
    return results


def n_squared(n, a, b):
    """
    Mathematical model for O(n²) complexity.
    
    Args:
        n: Input size (dataset size)
        a: Coefficient for n² term
        b: Constant term (overhead)
        
    Returns:
        Predicted execution time for input size n
    """
    return a * n**2 + b

def n_log_n(n, a, b):
    """
    Mathematical model for O(n log n) complexity.
    
    Args:
        n: Input size (dataset size)
        a: Coefficient for n*log(n) term
        b: Constant term (overhead)
        
    Returns:
        Predicted execution time for input size n
    """
    return a * n * np.log(n) + b

def n_plus_k(n, a, b):
    """
    Mathematical model for O(n+k) complexity where k is treated as a constant.
    
    Args:
        n: Input size (dataset size)
        a: Coefficient for n term
        b: Constant term (represents k, the value range)
        
    Returns:
        Predicted execution time for input size n
    """
    return a * n + b


def fit_complexity(
    sizes: List[int],
    times: List[float],
    complexity_type: str
) -> Tuple[np.ndarray, Callable, str]:
    """
    Fits empirical timing data to theoretical complexity models.
    
    Args:
        sizes (List[int]): Dataset sizes
        times (List[float]): Measured execution times
        complexity_type (str): Theoretical complexity to fit ("O(n²)", "O(n log n)", or "O(n+k)")
        
    Returns:
        Tuple[np.ndarray, Callable, str]: 
            - Optimized parameters from curve fitting
            - Fitted function that can predict execution time
            - Human-readable formula string
        
    Algorithm:
        1. Convert lists to NumPy arrays for mathematical operations
        2. Select appropriate model function based on complexity_type
        3. Use scipy's curve_fit to find optimal parameters
        4. Create a lambda function with the fitted parameters
        5. Generate a human-readable formula string
    """
    # Convert Python lists to NumPy arrays for efficient mathematical operations
    n = np.array(sizes)
    t = np.array(times)
    
    # Select the appropriate complexity model and fit the data
    if complexity_type == "O(n²)":
        # Fit data to quadratic model: T(n) = a*n² + b
        popt, _ = curve_fit(n_squared, n, t, maxfev=10000)
        # Create a function with the fitted parameters
        fitted_func = lambda x: n_squared(x, *popt)
        # Generate human-readable formula string with scientific notation
        formula = f"T(n) = {popt[0]:.2e}·n² + {popt[1]:.2e}"
    elif complexity_type == "O(n log n)":
        # Fit data to n*log(n) model: T(n) = a*n*log(n) + b
        popt, _ = curve_fit(n_log_n, n, t, maxfev=10000)
        # Create a function with the fitted parameters
        fitted_func = lambda x: n_log_n(x, *popt)
        # Generate human-readable formula string with scientific notation
        formula = f"T(n) = {popt[0]:.2e}·n·log(n) + {popt[1]:.2e}"
    elif complexity_type == "O(n+k)":
        # Fit data to linear model: T(n) = a*n + b (where b represents k, the value range)
        popt, _ = curve_fit(n_plus_k, n, t, maxfev=10000)
        # Create a function with the fitted parameters
        fitted_func = lambda x: n_plus_k(x, *popt)
        # Generate human-readable formula string with scientific notation
        formula = f"T(n) = {popt[0]:.2e}·n + {popt[1]:.2e}"
    else:
        # Handle invalid complexity type
        raise ValueError(f"Unknown complexity type: {complexity_type}")
    
    # Return optimized parameters, fitted function, and formula string
    return popt, fitted_func, formula


def plot_results(
    all_results: dict[str, List[Tuple[int, int, float]]],
    complexity_map: dict[str, str],
    plots_dir: str
) -> Tuple[dict[str, str], dict[str, str]]:
    """
    Generates performance plots for all sorting algorithms with regression curves.
    
    Args:
        all_results (dict[str, List[Tuple[int, int, float]]]): Benchmark results for each algorithm
        complexity_map (dict[str, str]): Maps algorithm names to their theoretical complexity
        plots_dir (str): Directory to save the generated plots
        
    Returns:
        Tuple[dict[str, str], dict[str, str]]: 
            - Dictionary mapping algorithm names to plot file paths
            - Dictionary mapping algorithm names to regression formulas
            
    Algorithm:
        1. Create output directory if it doesn't exist
        2. For each algorithm:
           a. Extract dataset sizes and execution times
           b. Fit complexity model to the data
           c. Generate smooth curve for visualization
           d. Create plot with actual data points and regression curve
           e. Save plot to file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize dictionaries to store plot paths and regression formulas
    plot_paths = {}
    formulas = {}
    
    # Process each algorithm's results
    for name, results in all_results.items():
        # Extract dataset sizes and execution times from results
        # results contains tuples of (dataset_index, dataset_size, execution_time)
        sizes = [r[1] for r in results]  # Dataset sizes
        times = [r[2] for r in results]  # Execution times
        
        # Get the theoretical complexity type for this algorithm
        # Default to O(n²) if not specified
        complexity_type = complexity_map.get(name, "O(n²)")
        
        # Fit the appropriate complexity model to the empirical data
        # Returns optimized parameters, fitted function, and formula string
        _, fitted_func, formula = fit_complexity(sizes, times, complexity_type)
        formulas[name] = formula
        
        # Generate smooth data points for the regression curve
        # This creates a visually appealing curve rather than connecting data points
        n_smooth = np.linspace(min(sizes), max(sizes), 200)
        t_smooth = fitted_func(n_smooth)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot actual data points as scatter plot
        plt.plot(sizes, times, marker="o", linestyle="", markersize=8, label="Actual Data", alpha=0.7)
        
        # Plot the fitted regression curve
        plt.plot(n_smooth, t_smooth, linestyle="-", linewidth=2, label=f"Regression: {complexity_type}", color="red")
        
        # Add axis labels and title
        plt.xlabel("Dataset Size", fontsize=12)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.title(f"{name} Performance", fontsize=14)
        
        # Add legend and grid for better visualization
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Generate filename and full file path
        filename = f"{name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(plots_dir, filename)
        
        # Save the plot to file and close the figure to free memory
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()
        
        # Store the plot path for return value
        plot_paths[name] = filepath
    
    # Return dictionaries of plot paths and regression formulas
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
