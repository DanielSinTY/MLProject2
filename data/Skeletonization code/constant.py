class WORM:
    IMAGE_SIZE1 = 750
    IMAGE_SIZE2 = 500
    INF = 1.0E10
    PI = 3.14159265359


class BW:
    SIDE_WIDTH = 4
    BOUNDARY_WIDTH = 2.1
    MINIMUM_HOLE_PROPORTION = 0.02
    BINARY_THRESHOLD = None
    LAPLACIAN_THRES = -1.2
    RAP_THRESHOLD = 0.69
    THRESHOLD_AREA_PROPORTION = 0.3
    MAX_EDGE_POINT_NUM = 2048
    SHRINK_LAPLACE_THRESHOLD = -8
    SHRINK_MIN_WIDTH = 3.0
    SHRINK_COMPENSATION_WIDTH = 2.0


class SKELETONIZE:
    POINT_NUM_MAX = 1000
    DEGREE_MAX = 5
    STORAGE_MAX = 200
    METRICS_MAX = 18
    ANGLE_THRESHOLD_NAN = WORM.PI / 3
    ALPHA = 2
    ANGLE_ERROR = 1E-6
    WORM_SPEED = 7
    WORM_TURNING_ANGLE = WORM.PI / 3
    FORWARD_DIST_PORTION = 3


class ROOT_SMOOTH:
    PARTITION_NUM = 100
    SMOOTH_DETAIL_LEVEL = 1
    SMOOTH_SCOPE = 5
    ZERO_BOUND = 1.5
    MULTIPLIER = 2 * SMOOTH_DETAIL_LEVEL + 1
    FULL_PARTS = MULTIPLIER * (2 * SMOOTH_SCOPE + 1)
    HALF_PARTS = (FULL_PARTS - 1) // 2


class SimpleException(Exception):
    def __init__(self, message=""):
        super().__init__(message)
        self.message = message

    def add_message(self, new_message):
        self.message += new_message

    def get_message(self):
        return self.message


CACHE_DIR = "./cache_data/"
