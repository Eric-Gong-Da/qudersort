import argparse
import json
import random
from typing import List


def load_settings(path: str) -> dict:
    with open(path, "r") as f:
        settings = json.load(f)
    
    required = ["n", "initial_size", "step"]
    for key in required:
        if key not in settings:
            raise ValueError(f"Missing required setting: {key}")
    
    return settings


def generate_datasets(
    n: int,
    initial_size: int,
    step: int,
    min_value: int = 0,
    max_value: int = 1000
) -> List[List[int]]:
    datasets = []
    
    for i in range(n):
        size = initial_size + i * step
        dataset = [random.randint(min_value, max_value) for _ in range(size)]
        datasets.append(dataset)
    
    return datasets


def save_datasets(datasets: List[List[int]], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(datasets, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate random datasets for sorting benchmarks")
    parser.add_argument(
        "--settings",
        default="generator_settings.json",
        help="Path to settings JSON file"
    )
    parser.add_argument(
        "--output",
        default="data/sample_sets.json",
        help="Path to output dataset file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    settings = load_settings(args.settings)
    
    n = settings["n"]
    initial_size = settings["initial_size"]
    step = settings["step"]
    min_value = settings.get("min_value", 0)
    max_value = settings.get("max_value", 1000)
    
    print(f"Generating {n} datasets...")
    print(f"  Initial size: {initial_size}")
    print(f"  Step: {step}")
    print(f"  Value range: [{min_value}, {max_value}]")
    
    datasets = generate_datasets(n, initial_size, step, min_value, max_value)
    
    print(f"Saving to {args.output}...")
    save_datasets(datasets, args.output)
    
    print(f"Generated {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets):
        print(f"  Dataset {i}: {len(dataset)} elements")
    
    print("Done!")


if __name__ == "__main__":
    main()
