from typing import Iterable, List, TypeVar

T = TypeVar("T")


def bubble_sort(items: Iterable[T]) -> List[T]:
    result = list(items)
    n = len(result)
    
    if n < 2:
        return result
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        if not swapped:
            break
    
    return result
