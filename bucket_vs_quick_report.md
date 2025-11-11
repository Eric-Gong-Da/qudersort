# Bucket Sort vs Quick Sort Comparison Report

## Overview

This report compares the performance of **Bucket Sort** (O(n+k)) and **Quick Sort** (O(n log n)) 
across varying dataset sizes. For image processing scenarios (e.g., 8-bit grayscale images with 
pixel values 0–255), bucket sort with 256 buckets can be particularly efficient for histogram processing.

## Combined Performance Plot

![Bucket Sort vs Quick Sort](plots/bucket_vs_quick_comparison.png)

## Algorithm Details

### Bucket Sort

**Theoretical Complexity:** O(n+k)

**Regression Formula:**
```
T(n) = 2.33e-07·n + -1.36e-04
```

**Statistics:**
- Mean: 0.000330s
- Median: 0.000239s
- Max: 0.002829s
- Min: 0.000009s

**Data:** [Bucket Sort CSV](data/bucket_sort_comparison.csv)

### Quick Sort

**Theoretical Complexity:** O(n log n)

**Regression Formula:**
```
T(n) = 9.43e-08·n·log(n) + 1.43e-04
```

**Statistics:**
- Mean: 0.001614s
- Median: 0.001398s
- Max: 0.011374s
- Min: 0.000010s

**Data:** [Quick Sort CSV](data/quick_sort_comparison.csv)

## Analysis

- **Speed Ratio (Max):** Bucket Sort is ~4.0x **faster** than Quick Sort at largest dataset
- **Speed Ratio (Mean):** Bucket Sort is ~4.9x **faster** on average

### Key Observations

1. **Bucket Sort** demonstrates linear O(n+k) growth, outperforming Quick Sort for this use case
2. **Quick Sort** shows O(n log n) behavior but has higher constant factors
3. For uniformly distributed data (like random integers 0–1000), Bucket Sort excels
4. **Image Processing Use Case:** For 8-bit grayscale images (pixel values 0–255):
   - Use 256 buckets to sort pixels for histogram equalization
   - Brightness adjustment operations benefit from linear-time sorting
   - Bucket sort is ideal when the range is known and relatively small (k ≈ n or k << n²)

### When Bucket Sort Wins

- **Known, limited range:** Data range is fixed (e.g., 0–255, 0–1000)
- **Uniform distribution:** Values spread relatively evenly across the range
- **Large datasets:** O(n) scales better than O(n log n) as n increases
- **Image processing:** Pixel sorting, histogram operations, color quantization
