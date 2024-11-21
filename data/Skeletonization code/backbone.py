import math
import struct
import numpy as np

class Backbone:
    def __init__(self, root_max_length):
        self.length = 0
        self.size = root_max_length
        self.wormLength = 0.0
        self.cood = np.zeros((self.size, 2)) if self.size > 0 else None

    def __del__(self):
        del self.cood

    @classmethod
    def persistence(cls, obj_ptr, out_file):
        with open(out_file, 'wb') as file:
            file.write(struct.pack('i', obj_ptr.length))
            file.write(obj_ptr.cood[:obj_ptr.length].tobytes())

    @classmethod
    def anti_persistence(cls, obj_ptr, in_file):
        with open(in_file, 'rb') as file:
            obj_ptr.length = struct.unpack('i', file.read(4))[0]
            obj_ptr.size = obj_ptr.length
            obj_ptr.cood = np.zeros((obj_ptr.size, 2))
            file.readinto(obj_ptr.cood[:obj_ptr.length])
            obj_ptr.wormLength = cls.calculate_worm_length(obj_ptr)

    def calculate_worm_length(self):
        x1, y1 = self.cood[0]
        length = 0.0
        for i in range(1, self.length):
            x2, y2 = self.cood[i]
            length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            x1, y1 = x2, y2
        return length

    def update_worm_length(self):
        self.wormLength = self.calculate_worm_length()

    def reverse(self):
        for i in range(self.length // 2):
            self.cood[i], self.cood[self.length - i - 1] = self.cood[self.length - i - 1].copy(), self.cood[i].copy()

    def __copy__(self):
        new_backbone = Backbone(self.size)
        new_backbone.length = self.length
        new_backbone.cood[:self.length] = self.cood[:self.length].copy()
        new_backbone.wormLength = self.calculate_worm_length(self)
        return new_backbone

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def __eq__(self, other):
        if not isinstance(other, Backbone):
            return False
        return (self.length == other.length and
                self.size == other.size and
                np.array_equal(self.cood, other.cood) and
                self.wormLength == other.wormLength)

    def __str__(self):
        return f"Backbone(length={self.length}, size={self.size}, wormLength={self.wormLength}, cood={self.cood[:self.length]})"