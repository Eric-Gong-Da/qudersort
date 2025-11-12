"""
===========================================================
Bucket Sort Implementation
===========================================================

Author: Gong Da
Student ID: 11536511
Last Updated: 2025-11-12
Purpose: Implements the Bucket Sort algorithm with O(n+k) time complexity.
         This is a distribution sorting algorithm that works by distributing 
         the elements of an array into a number of buckets. Each bucket is 
         then sorted individually, either using a different sorting algorithm 
         or by recursively applying the bucket sorting algorithm.
===========================================================
"""

import math
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def bucket_sort(items: Iterable[T]) -> List[T]:
    """
    Sorts a collection of items using the Bucket Sort algorithm.
    
    Args:
        items (Iterable[T]): An iterable collection of comparable items to sort
        
    Returns:
        List[T]: A new list containing the sorted items
        
    Time Complexity: O(n + k) where n is the number of elements and k is the number of buckets
    Space Complexity: O(n + k) for the buckets
    
    Algorithm:
        1. Convert input to list and handle edge cases
        2. Determine the range of values in the input
        3. Create buckets based on the square root of n (common heuristic)
        4. Distribute elements into appropriate buckets based on value ranges
        5. Sort each bucket individually
        6. Concatenate all buckets to produce the final sorted array
    
    Note: This implementation works best when input is uniformly distributed over a range.
    Performance degrades when the value range is much larger than the number of elements.
    """
    # Convert iterable to list to allow indexing operations
    result = list(items)
    
    # Get the length of the list for various calculations
    n = len(result)
    
    # Handle edge cases: empty list or single element
    if n < 2:
        return result
    
    # Find the minimum and maximum values to determine the range
    # This is necessary to map values to appropriate buckets
    min_val = min(result)
    max_val = max(result)
    
    # If all elements are the same, the list is already sorted
    if min_val == max_val:
        return result
    
    # Determine the number of buckets to use
    # Using square root of n is a common heuristic that balances:
    # - Memory usage (fewer buckets)
    # - Distribution efficiency (more buckets)
    bucket_count = max(1, int(math.sqrt(n)))
    
    # Initialize empty buckets as a list of empty lists
    # Each bucket will hold elements that fall within a specific range
    buckets: List[List[T]] = [[] for _ in range(bucket_count)]
    
    # Calculate the range of values for bucket distribution
    range_val = max_val - min_val
    
    # Distribute elements into appropriate buckets
    # Each element is mapped to a bucket based on its relative position in the value range
    for item in result:
        # Calculate which bucket this item belongs to
        # Formula: normalized position * (bucket_count - 1)
        # This maps the value range [min_val, max_val] to bucket indices [0, bucket_count-1]
        index = int((item - min_val) / range_val * (bucket_count - 1))
        
        # Ensure index is within valid bounds (handles edge cases with floating point arithmetic)
        index = max(0, min(bucket_count - 1, index))
        
        # Add the item to its appropriate bucket
        buckets[index].append(item)
    
    # Sort each bucket individually
    # Using the built-in sort (Timsort) which is efficient for small arrays
    for bucket in buckets:
        bucket.sort()
    
    # Concatenate all buckets to produce the final sorted array
    # This preserves the global ordering since elements in earlier buckets
    # are guaranteed to be smaller than elements in later buckets
    output = []
    for bucket in buckets:
        output.extend(bucket)
    
    # Return the sorted list
    return output
