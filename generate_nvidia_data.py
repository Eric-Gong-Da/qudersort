import argparse
import json
from pathlib import Path
import yfinance as yf
import numpy as np


def fetch_nvidia_data(period: str = "10y") -> list:
    print(f"Fetching NVIDIA (NVDA) stock data for period: {period}...")
    nvda = yf.Ticker("NVDA")
    hist = nvda.history(period=period)
    
    if hist.empty:
        raise ValueError("Failed to fetch NVIDIA data")
    
    print(f"Fetched {len(hist)} trading days of data")
    return hist


def create_datasets_from_stock(hist, n: int = 200, initial_size: int = 100, step: int = 50) -> list:
    prices = hist['Close'].values
    volumes = hist['Volume'].values
    
    prices_scaled = (prices * 100).astype(int)
    volumes_scaled = (volumes / 1000000).astype(int)
    
    all_values = np.concatenate([prices_scaled, volumes_scaled])
    
    datasets = []
    for i in range(n):
        size = initial_size + i * step
        
        if size > len(all_values):
            indices = np.random.choice(len(all_values), size, replace=True)
        else:
            start_idx = np.random.randint(0, max(1, len(all_values) - size + 1))
            indices = np.arange(start_idx, start_idx + size) % len(all_values)
        
        dataset = all_values[indices].tolist()
        datasets.append(dataset)
    
    return datasets


def save_datasets(datasets: list, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(datasets, f, indent=2)
    print(f"Saved {len(datasets)} datasets to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate datasets from NVIDIA stock data")
    parser.add_argument(
        "--period",
        default="10y",
        help="Time period for stock data (1y, 2y, 5y, 10y, max)"
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
    parser.add_argument(
        "--output",
        default="data/nvidia_sets.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    hist = fetch_nvidia_data(args.period)
    
    print(f"Generating {args.n} datasets...")
    print(f"  Initial size: {args.initial_size}")
    print(f"  Step: {args.step}")
    
    datasets = create_datasets_from_stock(
        hist,
        n=args.n,
        initial_size=args.initial_size,
        step=args.step
    )
    
    save_datasets(datasets, args.output)
    
    print(f"Generated {len(datasets)} datasets from NVIDIA stock data:")
    for i in range(min(5, len(datasets))):
        print(f"  Dataset {i}: {len(datasets[i])} elements")
    if len(datasets) > 5:
        print(f"  ... and {len(datasets) - 5} more")
    
    print("Done!")


if __name__ == "__main__":
    main()
