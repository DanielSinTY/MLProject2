from constant import SKELETONIZE, WORM
from utils import SelectMinimum
import numpy as np
#import cv2

class CandidatePoints:
    def __init__(self):
        self.POINT_NUM_MAX = SKELETONIZE.POINT_NUM_MAX
        self.IMAGE_SIZE = WORM.IMAGE_SIZE1
        self.point_num = 0
        self.cood = np.zeros((self.POINT_NUM_MAX, 2), dtype=int)
        self.hash_table = np.zeros(self.IMAGE_SIZE + 1, dtype=int)
        self.current_line = 0
        self.range_x = [0, 0]
        self.range_y = [0, 0]

    def reset(self):
        self.point_num = 0
        self.current_line = 0
        self.hash_table = np.zeros(self.IMAGE_SIZE + 1, dtype=int)

    def get_point_num(self):
        return self.point_num

    def get_center(self, points):
        if not points:
            raise ValueError("Input should include at least one point")
        center = np.mean([self.cood[pt] for pt in points], axis=0)
        return center

    def get_point(self, index):
        return self.cood[index]

    def is_point_nearby(self, index1, index2):
        return np.all(np.abs(self.cood[index1] - self.cood[index2]) < 2)

    def query_points_nearby(self, base_points, nearby_points):
        if not base_points:
            raise ValueError("Input should include at least one point")
        
        nearby_points.clear()
        self.range_calc(base_points)
        start_index = self.hash_table[self.range_x[0]]
        end_index = self.hash_table[self.range_x[1] + 1]

        for i in range(start_index, end_index):
            if self.cood[i, 1] > self.range_y[1]:
                i = self.hash_table[self.cood[i, 0] + 1] - 1
                continue
            for j in base_points:
                if self.is_point_nearby(j, i):
                    nearby_points.append(i)
                    break

    def query_points_by_pointer(self, base_point, direct_vec):
        from math import tan, sqrt

        ANGLE_THRESHOLD_NAN_TAN = tan(SKELETONIZE.ANGLE_THRESHOLD_NAN)
        ALPHA = SKELETONIZE.ALPHA
        METRICS_MAX = SKELETONIZE.METRICS_MAX

        start_index = self.hash_table[max(int(base_point[0] - METRICS_MAX), 0)]
        end_index = self.hash_table[min(int(base_point[0] + METRICS_MAX) + 1, self.current_line)]

        metrics_min = SelectMinimum(METRICS_MAX, -1)

        for i in range(start_index, end_index):
            pt = self.cood[i]
            if pt[1] > base_point[1] - METRICS_MAX and pt[1] < base_point[1] + METRICS_MAX:
                direct_vec_temp = [pt[0] - base_point[0], pt[1] - base_point[1]]
                angle_diff = self.included_angle_tan(direct_vec, direct_vec_temp)

                if 0 <= angle_diff and angle_diff < ANGLE_THRESHOLD_NAN_TAN and np.dot(direct_vec, direct_vec_temp)>0:
                    dist = sqrt((pt[0] - base_point[0]) ** 2 + (pt[1] - base_point[1]) ** 2)
                    weighted_dist = dist * (1 + ALPHA * angle_diff)
                    metrics_min.renew(weighted_dist, i)

        return metrics_min.get_min_index()


    def included_angle_tan(self, vec_1, vec_2):
        cross_product = np.cross(vec_1, vec_2)
        dot_product = np.dot(vec_1, vec_2)
        if abs(dot_product) < 1E-5:
            return -1
        return abs(cross_product) / dot_product

    def add_line(self):
        self.hash_table[self.current_line+1] = self.point_num
        self.current_line += 1

    def add_point_to_line(self, y):
        if self.point_num >= self.POINT_NUM_MAX:
            raise MemoryError("Point number exceeds the maximum limit")
        self.cood[self.point_num] = [self.current_line, y]
        self.point_num += 1

    def range_calc(self, base_points):
        x_coords = [self.cood[pt][0] for pt in base_points]
        y_coords = [self.cood[pt][1] for pt in base_points]
        self.range_x = [min(x_coords) - 1, max(x_coords) + 1]
        self.range_y = [min(y_coords) - 1, max(y_coords) + 1]

    def persistence(self, out_file):
        with open(out_file, 'wb') as f:
            np.save(f, self.cood[:self.point_num], allow_pickle=False)

    def anti_persistence(self, in_file):
        with open(in_file, 'rb') as f:
            self.cood = np.load(f)
            self.point_num = len(self.cood)
            self.current_line = max(self.cood[:, 0]) if self.point_num else 0

    def get_point_str(self, points):
        return "   ".join(f"{self.cood[pt][0]} {self.cood[pt][1]}" for pt in points)

    def get_whole_str(self):
        return "   ".join(f"{pt[0]} {pt[1]}" for pt in self.cood[:self.point_num])
