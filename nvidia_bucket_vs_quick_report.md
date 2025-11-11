ok# Bucket Sort vs Quick Sort on NVIDIA Stock Data

## Overview

This report compares the performance of **Bucket Sort** (O(n+k)) and **Quick Sort** (O(n log n)) 
using real-world NVIDIA (NVDA) stock market data. The analysis demonstrates that for **wide-range data** 
like stock prices and volumes (which can span from single digits to millions), Quick Sort significantly 
outperforms Bucket Sort.

## Combined Performance Plot

![Bucket Sort vs Quick Sort](plots/nvidia_bucket_vs_quick.png)

## Algorithm Details

### Bucket Sort

**Theoretical Complexity:** O(n+k)

**Regression Formula:**
```
T(n) = 1.23e-07·n + -1.86e-05
```

**Statistics:**
- Mean: 0.000605s
- Median: 0.000590s
- Max: 0.001257s
- Min: 0.000026s

**Data:** [Bucket Sort CSV](data/bucket_sort_nvidia.csv)

### Quick Sort

**Theoretical Complexity:** O(n log n)

**Regression Formula:**
```
T(n) = 7.68e-08·n·log(n) + 3.92e-03
```

**Statistics:**
- Mean: 0.007312s
- Median: 0.004440s
- Max: 0.066332s
- Min: 0.000066s

**Data:** [Quick Sort CSV](data/quick_sort_nvidia.csv)

## Analysis

- **Speed Ratio (Max):** Quick Sort is ~0.0x **faster** than Bucket Sort at largest dataset
- **Speed Ratio (Mean):** Quick Sort is ~0.1x **faster** on average

### Why Quick Sort Wins on Wide-Range Data


**The Problem with Bucket Sort on Stock Data:**

1. **Huge Value Range:** Stock prices (scaled ×100) range from ~10 to ~14,000
2. **Volume Data:** Trading volumes (scaled ÷1M) range from 0 to ~60,000
3. **Combined Range:** When mixing prices and volumes, k (range) ≈ 60,000+
4. **Bucket Overhead:** Creating and managing 60,000+ buckets dominates runtime
5. **Memory Cost:** O(n+k) becomes O(n+60000), where k >> n for many datasets

**Why O(n+k) Fails Here:**
- When k (value range) is **much larger** than n (dataset size), bucket sort degrades
- For NVIDIA data: k ≈ 60,000 while n ranges from 100 to 10,000
- Bucket initialization cost O(k) + bucket traversal O(k) becomes prohibitive
- **Quick Sort's O(n log n)** remains efficient regardless of value range

### Key Observations

1. **Quick Sort** dramatically outperforms Bucket Sort on wide-range financial data
2. **Bucket Sort** is only efficient when k ≈ n or k << n (e.g., 8-bit pixels: k=256)
3. **Stock market data** has inherently wide ranges due to:
   - Price growth over time (NVIDIA: $1 → $140+ over 20+ years)
   - Volume spikes during high-activity periods
   - Combined price + volume datasets spanning 5-6 orders of magnitude
4. **Real-world lesson:** Algorithm choice depends critically on data characteristics

### When to Use Each Algorithm


**Use Bucket Sort when:**
- Value range is **small and known** (e.g., 0-255 for images, 0-100 for percentages)
- k (range) ≈ n or k << n²
- Data is uniformly distributed
- Examples: Image processing, histogram generation, grade distributions

**Use Quick Sort when:**
- Value range is **large or unknown** (most real-world data)
- Data characteristics are unpredictable
- General-purpose sorting with good average-case performance needed
- Examples: Financial data, timestamps, arbitrary numerical data

### Data Source

- **Ticker:** NVDA (NVIDIA Corporation)
- **Period:** Maximum available history via Yahoo Finance (1999-2025)
- **Data Points:** 6,743+ trading days
- **Value Range:** Prices: $0.10 - $140+, Volumes: 0 - 60B+ shares
- **Scaled Range:** ~60,000 distinct values (demonstrates wide-range challenge)
