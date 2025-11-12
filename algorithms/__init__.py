"""
===========================================================
Sorting Algorithms Package
===========================================================

Author: Gong Da
Student ID: 11536511
Last Updated: 2025-11-12
Purpose: This package provides implementations of three classical sorting algorithms:
         Bubble Sort, Quick Sort, and Bucket Sort. It serves as the core module for
         the sorting algorithm performance analysis project.
===========================================================
"""

from algorithms.bubble_sort import bubble_sort
from algorithms.quick_sort import quick_sort
from algorithms.bucket_sort import bucket_sort

__all__ = ["bubble_sort", "quick_sort", "bucket_sort"]
