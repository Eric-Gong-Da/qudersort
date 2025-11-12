"""
===========================================================
Assignment 2 Part ONE: Data Generation Facade
===========================================================

Author: Gong Da
Student ID: 11536511
Last Updated: 2025-11-12
Purpose: This is a facade script that encapsulates data generation functionality
         for the sorting algorithm analysis assignment. It provides a simplified
         interface to generate both synthetic random datasets and real-world 
         NVIDIA (NVDA) stock market data.
===========================================================
"""

import argparse
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import main as generate_synthetic_data
from generate_nvidia_data import main as generate_nvidia_data


def main():
    parser = argparse.ArgumentParser(
        description="Data Generation Facade for Sorting Algorithm Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asg2_V2_partONE.py --type synthetic --output data/sample_sets.json
  python asg2_V2_partONE.py --type nvidia --output data/nvidia_sets.json --period max
        """
    )
    
    parser.add_argument(
        "--type",
        choices=["synthetic", "nvidia"],
        default="synthetic",
        help="Type of data to generate: synthetic (random) or nvidia (stock data)"
    )
    
    # Arguments for synthetic data generation
    parser.add_argument(
        "--settings",
        default="generator_settings.json",
        help="Path to settings JSON file for synthetic data generation"
    )
    
    # Arguments for NVIDIA data generation
    parser.add_argument(
        "--period",
        default="10y",
        help="Time period for NVIDIA stock data (1y, 2y, 5y, 10y, max)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of datasets to generate"
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=100,
        help="Initial dataset size"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Size increment between datasets"
    )
    
    # Common arguments
    parser.add_argument(
        "--output",
        default="data/sample_sets.json",
        help="Path to output dataset file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (synthetic data only)"
    )
    
    args = parser.parse_args()
    
    print("=========================================")
    print("Assignment 2 Part ONE: Data Generation")
    print("=========================================")
    
    if args.type == "synthetic":
        print("Generating synthetic random datasets...")
        print(f"Settings file: {args.settings}")
        print(f"Output file: {args.output}")
        if args.seed is not None:
            print(f"Random seed: {args.seed}")
        
        # Prepare arguments for the generate_data script
        sys.argv = [
            'generate_data.py',
            '--settings', args.settings,
            '--output', args.output
        ]
        if args.seed is not None:
            sys.argv.extend(['--seed', str(args.seed)])
            
        generate_synthetic_data()
        
    elif args.type == "nvidia":
        print("Generating NVIDIA stock market datasets...")
        print(f"Period: {args.period}")
        print(f"Number of datasets: {args.n}")
        print(f"Initial size: {args.initial_size}")
        print(f"Step: {args.step}")
        print(f"Output file: {args.output}")
        
        # Prepare arguments for the generate_nvidia_data script
        sys.argv = [
            'generate_nvidia_data.py',
            '--period', args.period,
            '--n', str(args.n),
            '--initial-size', str(args.initial_size),
            '--step', str(args.step),
            '--output', args.output
        ]
        
        generate_nvidia_data()
    
    print("\nData generation complete!")


if __name__ == "__main__":
    main()