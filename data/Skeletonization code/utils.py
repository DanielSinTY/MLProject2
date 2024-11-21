import numpy as np
import os
import math
from enum import Enum

class WORM_SHAPE(Enum):
    Normal = 0
    Omega = 1
    Delta = 2
    Circle = 3

# External variables (global in context)
worm_shape = WORM_SHAPE.Normal
performance_frequency = None

def get_current_performance_count():
    return os.times()[4] 

def get_performance_frequency():
    global performance_frequency
    performance_frequency = os.sysconf(os.sysconf_names['SC_CLK_TCK'])

def get_elapse_time(last, current):
    global performance_frequency
    if performance_frequency is None:
        get_performance_frequency()
    return ((current - last) * 1000) / performance_frequency

def calculate_curve_clockwise(data):
    det = 0
    num = len(data)
    for i in range(num - 2):
        p0, p1, p2 = data[i], data[i + 1], data[i + 2]
        det += (p1[0] * p2[1] + p0[0] * p1[1] + p0[1] * p2[0]) - (p0[1] * p1[0] + p1[1] * p2[0] + p0[0] * p2[1])
    return det > 0

def compute_median(data):
    data_sorted = sorted(data)
    num = len(data_sorted)
    if num % 2 == 0:
        return (data_sorted[num // 2] + data_sorted[num // 2 - 1]) / 2.0
    else:
        return data_sorted[num // 2]

def compute_median_list(data):
    data_sorted = sorted(data)
    num = len(data_sorted)
    if num % 2 == 0:
        return (data_sorted[num // 2] + data_sorted[num // 2 - 1]) / 2.0
    else:
        return data_sorted[num // 2]

def point_dist_square(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def included_angle_tan(vec_1, vec_2):
    angle = [vec_2[0] * vec_1[1] - vec_2[1] * vec_1[0], vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]]
    if angle[0] < 0:
        angle[0] = -angle[0]
    return -1 if angle[1] <= 1E-5 else angle[0] / angle[1]

def calc_clockwise_angle(p0, p1, p2):
    angle_clockwise = math.atan2(p2[0] - p1[0], p2[1] - p1[1]) - math.atan2(p0[0] - p1[0], p0[1] - p1[1])
    if angle_clockwise < 0:
        angle_clockwise += 2 * math.pi
    return angle_clockwise

def binary_chop(ordered_array, element_to_locate, left=0, right=None):
    if right is None:
        right = len(ordered_array) - 1
    if element_to_locate < ordered_array[left] or element_to_locate > ordered_array[right]:
        return -1
    if element_to_locate == ordered_array[right]:
        return right
    while right - left > 1:
        mid = (left + right) // 2
        if ordered_array[mid] < element_to_locate:
            left = mid
        elif ordered_array[mid] > element_to_locate:
            right = mid
        else:
            return mid
    return left

def int2str(num):
    return str(num)

class SelectMinimum:
    def __init__(self, initial_val, initial_index):
        self.min_val = initial_val
        self.min_index = initial_index

    def renew(self, new_val, new_index):
        if new_val < self.min_val:
            self.min_val = new_val
            self.min_index = new_index

    def get_min_index(self):
        return self.min_index

    def get_min_val(self):
        return self.min_val

def save_mat_to_file(mat, file_dir):
    rows, cols = mat.shape
    with open(file_dir, 'wb') as file:
        file.write(rows.to_bytes(4, 'little'))
        file.write(cols.to_bytes(4, 'little'))
        for i in range(rows):
            file.write(mat[i].astype(np.float64).tobytes())
