import math
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def bucket_sort(items: Iterable[T]) -> List[T]:
    result = list(items)
    n = len(result)
    
    if n < 2:
        return result
    
    min_val = min(result)
    max_val = max(result)
    
    if min_val == max_val:
        return result
    
    bucket_count = max(1, int(math.sqrt(n)))
    buckets: List[List[T]] = [[] for _ in range(bucket_count)]
    
    range_val = max_val - min_val
    
    for item in result:
        index = int((item - min_val) / range_val * (bucket_count - 1))
        index = max(0, min(bucket_count - 1, index))
        buckets[index].append(item)
    
    for bucket in buckets:
        bucket.sort()
    
    output = []
    for bucket in buckets:
        output.extend(bucket)
    
    return output
