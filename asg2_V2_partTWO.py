"""
===========================================================
Assignment 2 Part TWO: Benchmark and Analysis Facade
===========================================================

Author: Gong Da
Student ID: 11536511
Last Updated: 2025-11-12
Purpose: This is a facade script that encapsulates benchmarking and analysis functionality
         for the sorting algorithm analysis assignment. It provides a simplified
         interface to benchmark sorting algorithms on generated datasets and produce
         performance reports with plots and statistics.
===========================================================
"""

import argparse
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark import main as run_benchmark
from benchmark_bucket_vs_quick import main as run_bucket_vs_quick_benchmark
from benchmark_nvidia import main as run_nvidia_benchmark
from benchmark_nvidia_bucket_vs_quick import main as run_nvidia_bucket_vs_quick_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and Analysis Facade for Sorting Algorithm Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asg2_V2_partTWO.py --dataset data/sample_sets.json
  python asg2_V2_partTWO.py --benchmark-type bucket-vs-quick
  python asg2_V2_partTWO.py --benchmark-type nvidia --dataset data/nvidia_sets.json
        """
    )
    
    parser.add_argument(
        "--benchmark-type",
        choices=["all", "bucket-vs-quick", "nvidia", "nvidia-bucket-vs-quick"],
        default="all",
        help="Type of benchmark to run"
    )
    
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
    
    print("=========================================")
    print("Assignment 2 Part TWO: Benchmark and Analysis")
    print("=========================================")
    
    if args.benchmark_type == "all":
        print("Running comprehensive benchmark on all algorithms...")
        print(f"Dataset: {args.dataset}")
        print(f"Plots directory: {args.plots_dir}")
        print(f"CSV directory: {args.csv_dir}")
        print(f"Report file: {args.report}")
        
        # Prepare arguments for the main benchmark script
        sys.argv = [
            'benchmark.py',
            args.dataset,
            '--plots-dir', args.plots_dir,
            '--csv-dir', args.csv_dir,
            '--report', args.report
        ]
        
        run_benchmark()
        
    elif args.benchmark_type == "bucket-vs-quick":
        print("Running Bucket Sort vs Quick Sort benchmark...")
        print(f"Dataset: {args.dataset}")
        print(f"Plots directory: {args.plots_dir}")
        print(f"CSV directory: {args.csv_dir}")
        print(f"Report file: {args.report}")
        
        # Prepare arguments for the bucket vs quick benchmark script
        sys.argv = [
            'benchmark_bucket_vs_quick.py',
            args.dataset,
            '--plots-dir', args.plots_dir,
            '--csv-dir', args.csv_dir,
            '--report', args.report
        ]
        
        run_bucket_vs_quick_benchmark()
        
    elif args.benchmark_type == "nvidia":
        print("Running benchmark on NVIDIA stock data...")
        print(f"Dataset: {args.dataset}")
        print(f"Plots directory: {args.plots_dir}")
        print(f"CSV directory: {args.csv_dir}")
        print(f"Report file: {args.report}")
        
        # Prepare arguments for the NVIDIA benchmark script
        sys.argv = [
            'benchmark_nvidia.py',
            args.dataset,
            '--plots-dir', args.plots_dir,
            '--csv-dir', args.csv_dir,
            '--report', args.report
        ]
        
        run_nvidia_benchmark()
        
    elif args.benchmark_type == "nvidia-bucket-vs-quick":
        print("Running Bucket Sort vs Quick Sort benchmark on NVIDIA stock data...")
        print(f"Dataset: {args.dataset}")
        print(f"Plots directory: {args.plots_dir}")
        print(f"CSV directory: {args.csv_dir}")
        print(f"Report file: {args.report}")
        
        # Prepare arguments for the NVIDIA bucket vs quick benchmark script
        sys.argv = [
            'benchmark_nvidia_bucket_vs_quick.py',
            args.dataset,
            '--plots-dir', args.plots_dir,
            '--csv-dir', args.csv_dir,
            '--report', args.report
        ]
        
        run_nvidia_bucket_vs_quick_benchmark()
    
    print("\nBenchmark and analysis complete!")


if __name__ == "__main__":
    main()