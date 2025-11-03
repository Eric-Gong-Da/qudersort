# Sorting Algorithm Benchmark Report


## Bubble Sort

**Theoretical Complexity:** O(n²)

**Regression Formula:**
```
T(n) = 1.85e-08·n² + 5.48e-03
```

**Statistics:**
- Mean: 0.104099s
- Median: 0.075054s
- Max: 0.308343s

**Data:** [Bubble Sort CSV](data/bubble_sort.csv)

![Bubble Sort Performance](plots/bubble_sort.png)

## Quick Sort

**Theoretical Complexity:** O(n log n)

**Regression Formula:**
```
T(n) = 6.91e-08·n·log(n) + -2.68e-05
```

**Statistics:**
- Mean: 0.001050s
- Median: 0.000993s
- Max: 0.002388s

**Data:** [Quick Sort CSV](data/quick_sort.csv)

![Quick Sort Performance](plots/quick_sort.png)

## Bucket Sort

**Theoretical Complexity:** O(n+k)

**Regression Formula:**
```
T(n) = 1.27e-07·n + -2.30e-06
```

**Statistics:**
- Mean: 0.000252s
- Median: 0.000258s
- Max: 0.000569s

**Data:** [Bucket Sort CSV](data/bucket_sort.csv)

![Bucket Sort Performance](plots/bucket_sort.png)