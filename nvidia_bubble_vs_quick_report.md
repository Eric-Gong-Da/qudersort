# Bubble Sort vs Quick Sort on NVIDIA Stock Data

## Overview

This report compares the performance of **Bubble Sort** (O(n²)) and **Quick Sort** (O(n log n)) 
using real-world NVIDIA (NVDA) stock market data spanning multiple years. The datasets are derived 
from historical closing prices and trading volumes, demonstrating sorting performance on realistic 
financial data patterns.

## Combined Performance Plot

![Bubble Sort vs Quick Sort](plots/nvidia_bubble_vs_quick.png)

## Algorithm Details

### Bubble Sort

**Theoretical Complexity:** O(n²)

**Regression Formula:**
```
T(n) = 1.53e-08·n² + 5.87e-02
```

**Statistics:**
- Mean: 0.581632s
- Median: 0.488469s
- Max: 1.961092s
- Min: 0.000066s

**Data:** [Bubble Sort CSV](data/bubble_sort_nvidia.csv)

### Quick Sort

**Theoretical Complexity:** O(n log n)

**Regression Formula:**
```
T(n) = 9.48e-08·n·log(n) + 3.78e-03
```

**Statistics:**
- Mean: 0.007974s
- Median: 0.004516s
- Max: 0.067225s
- Min: 0.000065s

**Data:** [Quick Sort CSV](data/quick_sort_nvidia.csv)

## Analysis

- **Speed Ratio (Max):** Bubble Sort is ~29.2x slower than Quick Sort at largest dataset
- **Speed Ratio (Mean):** Bubble Sort is ~72.9x slower on average

### Key Observations

1. **Bubble Sort** exhibits clear O(n²) quadratic growth, becoming impractical as dataset size increases
2. **Quick Sort** maintains efficient O(n log n) behavior, scaling well with larger financial datasets
3. The performance gap widens dramatically with dataset size, confirming theoretical complexity predictions
4. **Real-World Financial Data:** This benchmark uses NVIDIA stock data (prices and volumes), 
   representing realistic sorting scenarios in financial applications like:
   - Order book sorting in trading systems
   - Time-series analysis requiring sorted price sequences
   - Volume-weighted calculations on sorted data
   - Market data aggregation and filtering

### Data Source

- **Ticker:** NVDA (NVIDIA Corporation)
- **Period:** Maximum available history via Yahoo Finance
- **Data Points:** Closing prices (scaled ×100) and trading volumes (scaled ÷1M)
- **Use Case:** Demonstrates algorithm performance on real financial time-series data
