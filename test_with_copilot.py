# Given an unsorted array of integers, write a function to find the kth smallest element in the array. Can you discuss the time and space complexity of your solution?

def kth_smallest(arr, k):
    if k < 1 or k > len(arr):
        return None

    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    left, right = 0, len(arr) - 1
    while left <= right:
        pivot_index = partition(left, right)
        if pivot_index == k - 1:
            return arr[pivot_index]
        elif pivot_index < k - 1:
            left = pivot_index + 1
        else:
            right = pivot_index - 1
    return None