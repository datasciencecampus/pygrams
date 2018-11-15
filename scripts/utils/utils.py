
class Utils(object):

    @staticmethod
    def bisearch_csr(array, target, start, end):
        while start <= end:
            middle = (start + end) // 2
            midpoint = array[middle]
            if midpoint > target:
                end = middle - 1
            elif midpoint < target:
                start = middle + 1
            else:
                return middle, array[middle] == target
        return 0, False
