import time

def quickSort(list, pivot=0):

    if pivot > len(list) - 1:
        pivot = 0
    if len(list) <= 1:
        return list

    before = [x for x in list[:pivot]]
    after = [x for x in list[pivot + 1:]]

    smaller = [x for x in before if x < list[pivot]]
    larger = [x for x in before if x >= list[pivot]]

    smaller += [x for x in after if x < list[pivot]]
    larger += [x for x in after if x >= list[pivot]]

    return quickSort(smaller) + [list[pivot]] + quickSort(larger)

# Main Function
if __name__ == '__main__':
    lst = [2, 4, 5, 1, 12, 100, 2, 4, 6, 9, 10]
    start = time.time()
    print(quickSort(lst))
    end = time.time()
    print(end - start)
