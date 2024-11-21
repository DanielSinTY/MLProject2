import numpy as np
import cv2
import pickle

from backbone import Backbone
from candidate_point import CandidatePoints 
from candidate_point_detect import CandidatePointsDetect
from graph_builder import GraphBuilder
from graph_prune import GraphPrune
from root_smooth import RootSmooth
from graph import Graph
from head_tail_list import HeadTailList
from constant import WORM, ROOT_SMOOTH

BACKBONE_ALTER_RANGE = 0.1
MAX_WORM_LENGTH_ERROR = 100
WORM_WIDTH_ALTER_RANGE = 0.1
BACKBONE_ALTER_RANGE1 = 0.6


class SearchBackbone:
    def __init__(self, worm_full_width=float(WORM.INF), worm_area=100):
        
        self.candidate_points_detect = CandidatePointsDetect()
        self.skeletonize = GraphBuilder()
        self.graph_prune = GraphPrune()
        self.root_smooth = RootSmooth()

        self.candidate_center_points = CandidatePoints()
        self.skeleton_graph = Graph()
        self.backbone = Backbone(ROOT_SMOOTH.PARTITION_NUM + 1)
        self.temp_backbone = Backbone(ROOT_SMOOTH.PARTITION_NUM + 1)
        self.actual_backbone = Backbone(ROOT_SMOOTH.PARTITION_NUM + 1)
        self.current_backbone = Backbone(ROOT_SMOOTH.PARTITION_NUM + 1)
        self.last_backbone = Backbone(ROOT_SMOOTH.PARTITION_NUM + 1)
        self.head_tail_list = HeadTailList()

        self.length_error_count = 0
        self.frame_count = 0
        self.first_pic = True
        self.worm_full_width = worm_full_width
        self.worm_area = worm_area
        self.mean_worm_length = 0.0
        self.mean_worm_width = 0.0
        self.last_mean_worm_length = 0.0
        self.pic_num_str = ""

    def set_width(self, worm_full_width):
        self.worm_full_width = worm_full_width

    def set_area(self, worm_area):
        self.worm_area = worm_area

    def data_processing(self):
        self.temp_backbone.update_worm_length()
        self.actual_backbone = self.temp_backbone
        print(f"Current backbone length: {self.temp_backbone.wormLength}, Mean length: {self.mean_worm_length}")

        if self.frame_count > 0 and (
                abs(self.temp_backbone.wormLength - self.mean_worm_length) > self.mean_worm_length * BACKBONE_ALTER_RANGE):
            self.length_error_count += 1
            if self.length_error_count <= MAX_WORM_LENGTH_ERROR:
                print(
                    f"WARNING: Frame abandoned because the worm length changes too much. {self.mean_worm_length} -> {self.temp_backbone.wormLength}")
                self.temp_backbone = self.backbone
                return
                

        self.length_error_count = 0
        self.backbone = self.temp_backbone
        self.last_mean_worm_length = self.mean_worm_length
        self.head_tail_list.add_head_tail_points(self.backbone.cood[0], self.backbone.cood[-1])
        self.last_backbone = self.current_backbone
        self.current_backbone = self.backbone

        worm_widths = [self.candidate_points_detect.get_dist(coord[0], coord[1]) for coord in self.backbone.cood]
        self.worm_area = self.candidate_points_detect.get_area()
        worm_widths_sorted = sorted(worm_widths)
        self.worm_full_width = 2 * worm_widths_sorted[int(len(worm_widths_sorted) * 0.8)]

        update_width = True
        if self.frame_count > 0 and (
                abs(self.worm_full_width - self.mean_worm_width) > self.mean_worm_width * WORM_WIDTH_ALTER_RANGE):
            update_width = False

        if self.frame_count == 0:
            self.mean_worm_length = self.backbone.wormLength
            self.mean_worm_width = self.worm_full_width
        else:
            self.mean_worm_length = (self.mean_worm_length * self.frame_count + self.backbone.wormLength) / (
                        self.frame_count + 1)
            if update_width:
                self.mean_worm_width = (self.mean_worm_width * self.frame_count + self.worm_full_width) / (
                            self.frame_count + 1)
            self.worm_full_width = self.mean_worm_width

        self.frame_count += 1
        self.first_pic = False

    def persistence(self, obj, out_file):
        with open(out_file, 'wb') as file:
            pickle.dump(obj, file)

    def next_stage(self):
        #print(self.stage_words[self.current_stage])

        # Assuming CACHE_DIR and pic_num_str are defined somewhere
        # persist_fun_ptrs[current_stage](persist_obj_ptrs[current_stage], CACHE_DIR + cache_dir_strs[current_stage] + pic_num_str)
        if self.current_stage in self.persist_fun_ptrs:
            self.persist_fun_ptrs[self.current_stage](self.persist_obj_ptrs[self.current_stage], CACHE_DIR + self.cache_dir_strs[self.current_stage] + pic_num_str)

        # Move to the next stage
        current_index = self.stages.index(self.current_stage)
        self.current_stage = self.stages[(current_index + 1) % len(self.stages)]

    def search(self, image, pic_num_str="temp", pic_num=0):
        self.pic_num_str = pic_num_str
        self.skeleton_graph.reset()
        self.candidate_center_points.reset()
        #print("Pic:", pic_num_str)

        self.candidate_points_detect.detect_points(image, self.candidate_center_points, self.worm_full_width,
                                                   self.worm_area)
        #self.next_stage()

        self.skeletonize.convert_to_graph(self.candidate_center_points, self.skeleton_graph, pic_num_str)
        #self.next_stage()

        self.graph_prune.prune(self.skeleton_graph, self.temp_backbone, self.worm_full_width, self.first_pic, pic_num)
        #self.next_stage()

        self.root_smooth.interpolate_and_equal_divide(self.temp_backbone, ROOT_SMOOTH.PARTITION_NUM)
        #self.next_stage()

        self.data_processing()
        #self.next_stage()

        return self.backbone

    def length_error(self):
        return self.length_error_count > 0
    
    def save_centerline_results(self, filename):
        length_error = self.length_error()
        with open(filename, 'wb') as file:
            np.save(file, {
                'length_error': length_error,
                'last_mean_worm_length': self.last_mean_worm_length,
                'worm_full_width': self.worm_full_width,
                'actual_backbone': self.actual_backbone.cood,
                'current_backbone': self.current_backbone.cood,
                'last_backbone': self.last_backbone.cood
            })

    def initialize(self):
        self.length_error_count = 0
        self.frame_count = 0
        self.temp_backbone = self.backbone
