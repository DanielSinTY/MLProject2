import numpy as np
from constant import ROOT_SMOOTH

class RootSmooth:
    def __init__(self):
        self.origin_num = 0
        self.INTERPOLATE_NUM = 0
        self.interpolate_coodinate = None
        self.coodinate = None

    def interpolate(self):
        coefficent = np.zeros(ROOT_SMOOTH.FULL_PARTS)
        dh = ROOT_SMOOTH.ZERO_BOUND / (ROOT_SMOOTH.SMOOTH_SCOPE * ROOT_SMOOTH.MULTIPLIER)
        sum_array = np.zeros(ROOT_SMOOTH.MULTIPLIER)

        for i in range(1, ROOT_SMOOTH.HALF_PARTS + 1):
            value = np.exp(-dh * dh * i * i / 2)
            coefficent[ROOT_SMOOTH.HALF_PARTS - i] = value
            coefficent[ROOT_SMOOTH.HALF_PARTS + i] = value
            sum_array[(ROOT_SMOOTH.HALF_PARTS - i) % ROOT_SMOOTH.MULTIPLIER] += value
            sum_array[(ROOT_SMOOTH.HALF_PARTS + i) % ROOT_SMOOTH.MULTIPLIER] += value
        
        coefficent[ROOT_SMOOTH.HALF_PARTS] = 1
        sum_array[ROOT_SMOOTH.SMOOTH_DETAIL_LEVEL] += 1

        for i in range(ROOT_SMOOTH.FULL_PARTS):
            coefficent[i] /= sum_array[i % ROOT_SMOOTH.MULTIPLIER]

        self.interpolate_coodinate = np.zeros((self.INTERPOLATE_NUM, 2))

        for i in range(self.origin_num):
            for j in range(ROOT_SMOOTH.FULL_PARTS):
                index = i * ROOT_SMOOTH.MULTIPLIER + j - ROOT_SMOOTH.HALF_PARTS
                if 0 <= index < self.INTERPOLATE_NUM:
                    self.interpolate_coodinate[index] += self.coodinate[i] * coefficent[j]

        for i in range(ROOT_SMOOTH.SMOOTH_SCOPE):
            symmetry_cood = 2 * self.coodinate[0] - self.coodinate[i + 1]
            for j in range(ROOT_SMOOTH.FULL_PARTS):
                index = (-1 - i) * ROOT_SMOOTH.MULTIPLIER + j - ROOT_SMOOTH.HALF_PARTS
                if 0 <= index < self.INTERPOLATE_NUM:
                    self.interpolate_coodinate[index] += symmetry_cood * coefficent[j]

            symmetry_cood = 2 * self.coodinate[self.origin_num - 1] - self.coodinate[self.origin_num - i - 2]
            for j in range(ROOT_SMOOTH.FULL_PARTS):
                index = (self.origin_num + i) * ROOT_SMOOTH.MULTIPLIER + j - ROOT_SMOOTH.HALF_PARTS
                if 0 <= index < self.INTERPOLATE_NUM:
                    self.interpolate_coodinate[index] += symmetry_cood * coefficent[j]

    def equal_divide(self, partition_num):
        self.coodinate[0] = self.interpolate_coodinate[0]
        self.coodinate[partition_num] = self.interpolate_coodinate[self.INTERPOLATE_NUM - 1]

        length_each = np.zeros(self.INTERPOLATE_NUM)
        for i in range(1, self.INTERPOLATE_NUM):
            length_each[i] = length_each[i - 1] + np.sqrt(np.sum((self.interpolate_coodinate[i] - self.interpolate_coodinate[i - 1]) ** 2))
        
        full_length = length_each[-1]
        segment_length = full_length / partition_num

        for i in range(1, partition_num):
            index = self.binary_chop(length_each, segment_length * i, 0, self.INTERPOLATE_NUM - 1)
            alpha = (segment_length * i - length_each[index]) / (length_each[index + 1] - length_each[index])
            self.coodinate[i] = self.interpolate_coodinate[index] * (1 - alpha) + self.interpolate_coodinate[index + 1] * alpha

    def binary_chop(self, array, value, low, high):
        while low < high:
            mid = (low + high) // 2
            if array[mid] < value:
                low = mid + 1
            else:
                high = mid
        return low

    def interpolate_and_equal_divide(self, centerline_to_smooth, partition_num):
        self.origin_num = centerline_to_smooth.length
        print(self.origin_num)
        self.INTERPOLATE_NUM = self.origin_num * ROOT_SMOOTH.MULTIPLIER - 2
        self.coodinate = centerline_to_smooth.cood

        self.interpolate_coodinate = np.zeros((self.INTERPOLATE_NUM, 2))
        self.interpolate()

        self.coodinate = np.zeros((partition_num + 1, 2))
        self.equal_divide(partition_num)

        centerline_to_smooth.cood = self.coodinate
        centerline_to_smooth.length = partition_num + 1
        centerline_to_smooth.size = partition_num + 1
