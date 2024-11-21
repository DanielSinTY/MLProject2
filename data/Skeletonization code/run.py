import os
import cv2
import numpy as np
import pickle
import time

from search_backbone import SearchBackbone
from backbone import Backbone
from utils import int2str, load_skip_list, image_get, in_skip_list

INITIAL_WORM_AREA = 1800
PerformanceFrequency = 0  
wormShape = 'Normal'

PIC_START = 0
PIC_END = 0

pic_num_prefix = ""
file_path = ""
backbone_path = ""
skiplist_path = ""
SkipList_Num = 0

def load_skip_list(filename):
    skip_list = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                skip_list.append(int(line.strip()))
    except IOError:
        pass
    return skip_list

def image_get(file_path, img_index):
    image_filename = f"{file_path}/{pic_num_prefix}{int2str(img_index)}.tiff"
    print(image_filename)
    image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape[0] != WORM.IMAGE_SIZE1 or image.shape[1] != WORM.IMAGE_SIZE2:
        print(f"Pic {int2str(img_index)} : Not Exist or Wrong Image Size(run)!")
        return None
    return image

def in_skip_list(index, skip_list):
    return index in skip_list

def main(file_path, backbone_path, pic_start, pic_end):
    global PIC_START, PIC_END, skiplist_path, skip_list, search_backbone

    PIC_START = pic_start
    PIC_END = pic_end
    skiplist_path = os.path.join(backbone_path, "skiplist.txt")

    # Load skip list
    skip_list = load_skip_list(skiplist_path)

    search_backbone = SearchBackbone()

    for pic_num in range(PIC_START, PIC_END + 1):
        image = image_get(file_path, pic_num)
        if image is not None and not in_skip_list(pic_num, skip_list):
            filename = f"{backbone_path}/backbone_{pic_num}.bin"

            try:
                start = time.time()
                backbone = search_backbone.search(image)
                end = time.time()
                print(f"Centerline extraction time consumption: {(end - start) * 1000}ms")

                with open(filename, 'wb') as file:
                    pickle.dump(backbone, file)
            except Exception as e:
                print(f"Error: {e}")
            search_backbone.save_centerline_results(filename)  # save backbone results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: script.py <file_path> <backbone_path> <pic_start> <pic_end>")
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
