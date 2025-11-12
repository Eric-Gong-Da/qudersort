"""
===========================================================
Quick Sort Implementation
===========================================================

Author: Gong Da
Student ID: 11536511
Last Updated: 2025-11-12
Purpose: Implements the Quick Sort algorithm with O(n log n) average time complexity.
         This is a divide-and-conquer algorithm that works by selecting a 'pivot' element 
         from the array and partitioning the other elements into two sub-arrays, 
         according to whether they are less than or greater than the pivot.
===========================================================
"""

from typing import Iterable, List, TypeVar

T = TypeVar("T")


def quick_sort(items: Iterable[T]) -> List[T]:
    """
    Sorts a collection of items using the Quick Sort algorithm.
    
    Args:
        items (Iterable[T]): An iterable collection of comparable items to sort
        
    Returns:
        List[T]: A new list containing the sorted items
        
    Time Complexity: O(n log n) average case, O(n²) worst case
    Space Complexity: O(log n) auxiliary space (due to recursion stack)
    
    Algorithm:
        1. Convert input to list to allow indexing
        2. Handle edge cases (empty or single element)
        3. Apply in-place quicksort to the list
        4. Return the sorted list
    """
    # Convert iterable to list to allow indexing operations
    result = list(items)
    
    # Handle edge cases: empty list or single element
    if len(result) < 2:
        return result
    
    # Apply in-place quicksort to the list
    # Start with the entire array (indices 0 to len(result) - 1)
    _quick_sort_inplace(result, 0, len(result) - 1)
    
    # Return the sorted list
    return result


def _quick_sort_inplace(arr: List[T], low: int, high: int) -> None:
    """
    Recursively sorts a subarray in-place using the Quick Sort algorithm.
    
    Args:
        arr (List[T]): The list to sort
        low (int): Starting index of the subarray to sort
        high (int): Ending index of the subarray to sort
        
    Returns:
        None: Sorts the array in-place
        
    Algorithm:
        1. Base case: if low >= high, subarray has 0 or 1 elements (already sorted)
        2. Partition the subarray around a pivot element
        3. Recursively sort the subarrays on both sides of the pivot
    """
    # Base case: if low >= high, subarray has 0 or 1 elements (already sorted)
    if low < high:
        # Partition the subarray and get the pivot index
        # After partitioning, pivot is in its final sorted position
        pivot_idx = _partition(arr, low, high)
        
        # Recursively sort the left subarray (elements less than pivot)
        _quick_sort_inplace(arr, low, pivot_idx - 1)
        
        # Recursively sort the right subarray (elements greater than pivot)
        _quick_sort_inplace(arr, pivot_idx + 1, high)


def _partition(arr: List[T], low: int, high: int) -> int:
    """
    Partitions a subarray around a pivot element.
    
    Args:
        arr (List[T]): The list to partition
        low (int): Starting index of the subarray to partition
        high (int): Ending index of the subarray to partition
        
    Returns:
        int: The final index of the pivot element
        
    Algorithm:
        1. Select the last element as the pivot
        2. Initialize i to low - 1 (index of smaller element)
        3. For each element from low to high-1:
           a. If element <= pivot, increment i and swap elements at i and j
        4. Place pivot in its correct position by swapping with element at i+1
        5. Return the pivot's final index
        
    This implementation uses the Lomuto partition scheme.
    """
    # Select the last element as the pivot
    pivot = arr[high]
    
    # Initialize i to low - 1 (index of smaller element)
    # Elements at indices <= i will be <= pivot
    i = low - 1
    
    # Traverse elements from low to high-1
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            # Increment index of smaller element
            i += 1
            # Swap elements at positions i and j
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in its correct position by swapping with element at i+1
    # After this swap, all elements <= pivot are to the left of the pivot
    # and all elements > pivot are to the right of the pivot
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    # Return the pivot's final index
    return i + 1
