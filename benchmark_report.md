# Sorting Algorithm Benchmark Report


## Bubble Sort

**Theoretical Complexity:** O(n²)

**Regression Formula:**
```
T(n) = 1.92e-08·n² + -8.75e-04
```

**Statistics:**
- Mean: 0.101538s
- Median: 0.075295s
- Max: 0.306443s

**Data:** [Bubble Sort CSV](results/bubble_sort.csv)

![Bubble Sort Performance](plots/bubble_sort.png)

## Quick Sort

**Theoretical Complexity:** O(n log n)

**Regression Formula:**
```
T(n) = 6.91e-08·n·log(n) + -2.48e-05
```

**Statistics:**
- Mean: 0.001052s
- Median: 0.000998s
- Max: 0.002314s

**Data:** [Quick Sort CSV](results/quick_sort.csv)

![Quick Sort Performance](plots/quick_sort.png)

## Bucket Sort

**Theoretical Complexity:** O(n+k)

**Regression Formula:**
```
T(n) = 1.32e-07·n + -8.45e-06
```

**Statistics:**
- Mean: 0.000256s
- Median: 0.000247s
- Max: 0.001559s

**Data:** [Bucket Sort CSV](results/bucket_sort.csv)

![Bucket Sort Performance](plots/bucket_sort.png)