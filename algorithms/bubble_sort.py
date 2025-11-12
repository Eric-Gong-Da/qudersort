"""
===========================================================
Bubble Sort Implementation
===========================================================

Author: Gong Da
Student ID: 11536511
Last Updated: 2025-11-12
Purpose: Implements the Bubble Sort algorithm with O(n²) time complexity.
         This is a simple comparison-based sorting algorithm that repeatedly 
         steps through the list, compares adjacent elements and swaps them 
         if they are in the wrong order.
===========================================================
"""

from typing import Iterable, List, TypeVar

T = TypeVar("T")


def bubble_sort(items: Iterable[T]) -> List[T]:
    """
    Sorts a collection of items using the Bubble Sort algorithm.
    
    Args:
        items (Iterable[T]): An iterable collection of comparable items to sort
        
    Returns:
        List[T]: A new list containing the sorted items
        
    Time Complexity: O(n²) in worst and average cases, O(n) in best case (already sorted)
    Space Complexity: O(1) auxiliary space (in-place sorting with only a few variables)
    
    Algorithm:
        1. Convert input to list to allow indexing
        2. For each position i from 0 to n-1:
           a. Compare adjacent elements from 0 to n-i-1
           b. Swap if they are in wrong order
           c. If no swaps occurred, the list is sorted (early termination)
    """
    # Convert iterable to list to allow indexing operations
    result = list(items)
    
    # Get the length of the list for loop bounds
    n = len(result)
    
    # Handle edge cases: empty list or single element
    if n < 2:
        return result
    
    # Outer loop: controls how many passes through the array we make
    # After i passes, the last i elements are in their final sorted positions
    for i in range(n):
        # Flag to detect if any swaps were made in this pass
        # If no swaps occur, the array is already sorted
        swapped = False
        
        # Inner loop: performs comparisons and swaps on adjacent elements
        # Range decreases by i each iteration since the last i elements are sorted
        for j in range(0, n - i - 1):
            # Compare adjacent elements and swap if they're in wrong order
            if result[j] > result[j + 1]:
                # Swap elements using Python's tuple unpacking
                result[j], result[j + 1] = result[j + 1], result[j]
                # Mark that a swap occurred
                swapped = True
        
        # Early termination optimization: if no swaps occurred,
        # the array is already sorted, so we can exit early
        if not swapped:
            break
    
    # Return the sorted list
    return result
