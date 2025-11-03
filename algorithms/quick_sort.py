from typing import Iterable, List, TypeVar

T = TypeVar("T")


def quick_sort(items: Iterable[T]) -> List[T]:
    result = list(items)
    if len(result) < 2:
        return result
    
    _quick_sort_inplace(result, 0, len(result) - 1)
    return result


def _quick_sort_inplace(arr: List[T], low: int, high: int) -> None:
    if low < high:
        pivot_idx = _partition(arr, low, high)
        _quick_sort_inplace(arr, low, pivot_idx - 1)
        _quick_sort_inplace(arr, pivot_idx + 1, high)


def _partition(arr: List[T], low: int, high: int) -> int:
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
