import time

def quickSort(lst):
    if len(lst) <= 1:
        return lst
    smaller = [x for x in lst[1:] if x < lst[0]]
    larger = [x for x in lst[1:] if x >= lst[0]]
    return quickSort(smaller) + [lst[0]] + quickSort(larger)


# Main Function
if __name__ == '__main__':
    lst = [2, 4, 5, 1, 12, 100, 2, 4, 6, 9, 10]
    start = time.time()
    print(quickSort(lst))
    end = time.time()
    print(end - start)
