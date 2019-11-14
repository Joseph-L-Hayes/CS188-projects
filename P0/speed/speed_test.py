import quickSort
import quickSort2
import time
import math

qsWins = 0
me = 0
lst = [2, 4, 5, 1, 12, 100, 2, 4, 6, 9, 10]

for x in range(50000):
    start = time.time()
    quickSort.quickSort(lst)
    end = time.time()
    total = end - start
    # print(end - start)

    # lst = [2, 4, 5, 1, 12, 100, 2, 4, 6, 9, 10]

    start2 = time.time()
    quickSort2.quickSort(lst,3)
    end2 = time.time()
    total2 = end2 - start2

    if total >= total2:
        qsWins += 1

    else:
        me += 1

print('QS won %i times' % qsWins)
print('You won %i times' % me)
