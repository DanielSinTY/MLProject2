import numpy as np
import cv2
from constant import SKELETONIZE, BW, WORM
from candidate_point import CandidatePoints
import matplotlib.pyplot as plt

class CandidatePointsDetect:
    HALF_THRES = BW.LAPLACIAN_THRES / 2

    def __init__(self):
        self.distance_matrix = np.zeros((WORM.IMAGE_SIZE1, WORM.IMAGE_SIZE2), dtype=np.float32)
        self.laplacian_matrix = np.zeros_like(self.distance_matrix)
        self.worm_xy_range = [0, 0, 0, 0]

    def distance_retrace(self, width_max):
        RAP_SLOPE = BW.RAP_THRESHOLD / (1 - BW.RAP_THRESHOLD)
        mask = self.distance_matrix > BW.RAP_THRESHOLD * width_max
        self.distance_matrix[mask] = np.mod(width_max - self.distance_matrix[mask], width_max) * RAP_SLOPE

    def calc_lapmat_of_inner_part(self):
        for i in range(1, self.distance_matrix.shape[0] - 1):
            for j in range(1, self.distance_matrix.shape[1] - 1):
                if self.distance_matrix[i, j] > 2:
                    self.laplacian_matrix[i, j] = (
                        self.distance_matrix[i - 1, j] + self.distance_matrix[i + 1, j] +
                        self.distance_matrix[i, j - 1] + self.distance_matrix[i, j + 1] -
                        4 * self.distance_matrix[i, j]
                    )
                    if self.laplacian_matrix[i, j] > 0:
                        self.laplacian_matrix[i, j] = 0
                else:
                    self.laplacian_matrix[i, j] = 0

        # Set the boundary of the laplacian matrix to zero
        self.laplacian_matrix[:, 0] = self.laplacian_matrix[:, -1] = 0
        self.laplacian_matrix[0, :] = self.laplacian_matrix[-1, :] = 0

    def lap_value_small_enough(self, i, j):
        if self.laplacian_matrix[i, j] < BW.LAPLACIAN_THRES:
            return True
        if self.laplacian_matrix[i, j] >= self.HALF_THRES:
            return False
        surrounding = sum(
            self.laplacian_matrix[i + dx, j + dy] < self.HALF_THRES
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == dy == 0)
        )
        return surrounding >= 4

    def catch_candidate_by_lapmat(self, candidate_points: CandidatePoints):
        for _ in range(self.worm_xy_range[2]):
            candidate_points.add_line()

        for i in range(1, self.distance_matrix.shape[0] - 1):
            candidate_points.add_line()
            for j in range(1, self.distance_matrix.shape[1] - 1):
                if self.lap_value_small_enough(i, j):
                    candidate_points.add_point_to_line(j + self.worm_xy_range[0] + 1)
        candidate_points.add_line()
        candidate_points.add_line()

    def contour_range_get(self, contours, contour_idx):
        c = contours[contour_idx]
        x_min, x_max = np.min(c[:, :, 0]), np.max(c[:, :, 0])
        y_min, y_max = np.min(c[:, :, 1]), np.max(c[:, :, 1])
        self.worm_xy_range = [
            max(x_min - BW.SIDE_WIDTH, 1),
            min(x_max + BW.SIDE_WIDTH, WORM.IMAGE_SIZE1),
            max(y_min - BW.SIDE_WIDTH, 1),
            min(y_max + BW.SIDE_WIDTH, WORM.IMAGE_SIZE1)
        ]
        
    def denoise_and_worm_locate(self, area):
        contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cont) - sum(cv2.contourArea(contours[i]) for i in range(len(contours)) if hierarchy[0][i][3] == idx) for idx, cont in enumerate(contours)]
        

        area_diffs = [abs(area - a) for a in areas]
        selected_index = np.argmin(area_diffs)
        selected_area = areas[selected_index]

        if selected_index == -1:
            raise Exception("Can't get connected components of the worm")

        self.area = selected_area
        self.contour_range_get(contours, selected_index)

        cv2.drawContours(self.binary_image, contours, selected_index, (255), cv2.FILLED, hierarchy=hierarchy, maxLevel=1)
        self.binary_image = self.binary_image[self.worm_xy_range[2]:self.worm_xy_range[3], self.worm_xy_range[0]:self.worm_xy_range[1]]
       
        # Invert to fill holes
        self.binary_image = cv2.bitwise_not(self.binary_image)
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            hole_area = cv2.contourArea(contours[i])
            if hole_area < self.area * BW.MINIMUM_HOLE_PROPORTION:
                cv2.drawContours(self.binary_image, contours, i, (255), cv2.FILLED)
        self.binary_image = cv2.bitwise_not(self.binary_image)

        # Set image boundary to 0
        self.binary_image[:, 0] = 0
        self.binary_image[:, -1] = 0
        self.binary_image[0, :] = 0
        self.binary_image[-1, :] = 0
        
    def adaptive_threshold(self, image, area):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        cum_hist = np.cumsum(hist[::-1])[::-1]
        threshold_index = np.searchsorted(cum_hist, area * BW.THRESHOLD_AREA_PROPORTION)
        return threshold_index + 3 if threshold_index < 256 else 255

    def shrink_width(self, worm_region):
        # Calculate the shrink width of the worm body
        worm_region = self.binary_image[self.worm_xy_range[2]:self.worm_xy_range[3], self.worm_xy_range[0]:self.worm_xy_range[1]]
        self.distance_matrix = cv2.distanceTransform(worm_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        laplacian_threshold = BW.SHRINK_LAPLACE_THRESHOLD
        mask = (worm_region > 0) & (self.laplacian_matrix < laplacian_threshold)
        eligible_widths = self.distance_matrix[mask]
        median_width = np.median(eligible_widths)
        return max(median_width - BW.SHRINK_COMPENSATION_WIDTH, BW.SHRINK_MIN_WIDTH)

    def worm_shrink(self, worm_image, area):
        self.worm_region = worm_image[self.worm_xy_range[2]:self.worm_xy_range[3], self.worm_xy_range[0]:self.worm_xy_range[1]]
        shrink_width = self.shrink_width(self.worm_region)
        self.binary_image = self.distance_matrix > shrink_width

    def distance_modification(self, width):
        self.adhesion_width = BW.RAP_THRESHOLD * width
        for i in range(1, self.distance_matrix.shape[0] - 1):
            for j in range(1, self.distance_matrix.shape[1] - 1):
                if (self.distance_matrix[i, j] > self.adhesion_width and
                        self.laplacian_matrix[i, j] < BW.LAPLACIAN_THRES):
                    self.binary_image[i, j] = 0

    def image_dilate(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(img, kernel)

    def image_erode(self, img):
        for i in range(2, self.binary_image.shape[0] - 2):
            for j in range(2, self.binary_image.shape[1] - 2):
                if (self.binary_image[i, j] and
                        (np.sum(self.binary_image[i - 1:i + 2, j - 1:j + 2]) < 8)):
                    img[i, j] = 0

    def image_fillhole(self, img, area):
        img = cv2.bitwise_not(img)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            hole_area = cv2.contourArea(contours[i])
            retain_hole = hole_area > area * BW.MINIMUM_HOLE_PROPORTION / 2
            cv2.drawContours(img, contours, i, 255 * retain_hole, cv2.FILLED, hierarchy, 0)

        img = cv2.bitwise_not(img)
        img[:, [0, -1]] = 0
        img[[0, -1], :] = 0
        return img

    def save2file(self, binary_cache_dir, dist_cache_dir, lap_cache_dir, pic_num):
        with open(binary_cache_dir + pic_num, 'wb') as file:
            rows, cols = self.binary_image.shape
            file.write(self.worm_xy_range.astype('int32').tobytes())
            file.write(rows.to_bytes(4, byteorder='little'))
            file.write(cols.to_bytes(4, byteorder='little'))
            file.write(self.binary_image.tobytes())

        np.save(dist_cache_dir + pic_num, self.distance_matrix_modification)
        with open(lap_cache_dir + pic_num, 'wb') as lap_file:
            rows, cols = self.distance_matrix.shape
            lap_file.write(rows.to_bytes(4, byteorder='little'))
            lap_file.write(cols.to_bytes(4, byteorder='little'))
            lap_file.write(self.laplacian_matrix.tobytes())

    def get_dist(self, x, y):
        x -= self.worm_xy_range[2]
        y -= self.worm_xy_range[0]
        if x < 0 or y < 0 or x >= self.distance_matrix.shape[0] or y >= self.distance_matrix.shape[1]:
            return 0
        alpha = x - int(x)
        beta = y - int(y)
        x0 = int(x)
        y0 = int(y)
        return (self.distance_matrix[x0, y0] * (1 - alpha) * (1 - beta) +
                self.distance_matrix[x0 + 1, y0] * alpha * (1 - beta) +
                self.distance_matrix[x0, y0 + 1] * (1 - alpha) * beta +
                self.distance_matrix[x0 + 1, y0 + 1] * alpha * beta)

    def detect_points(self, image, candidate_points, width, area):
        candidate_points.reset()
        Binary_Threshold = 0
        self.binary_image = (image > Binary_Threshold).astype(np.uint8)
        
        self.denoise_and_worm_locate(area)
        

        self.distance_matrix = cv2.distanceTransform(self.binary_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.calc_lapmat_of_inner_part()
        self.distance_modification(width)
        self.distance_matrix_modification = cv2.distanceTransform(self.binary_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.calc_lapmat_of_inner_part()
        self.catch_candidate_by_lapmat(candidate_points)
        
    def get_area(self):
        return self.area
