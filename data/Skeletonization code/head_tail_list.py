import numpy as np

class HeadTailList:
    def __init__(self):
        self.head = np.zeros((3, 2))
        self.tail = np.zeros((3, 2))
        self.current_index = 0
        self.last_index = 0
        self.available = False

    def is_available(self):
        return self.available

    def add_head_tail_points(self, h, t):
        # Add head and tail points to the respective lists, manage indices and availability
        self.head[self.current_index] = h
        self.tail[self.current_index] = t
        self.last_index = self.current_index
        self.current_index += 1
        if not self.available and self.current_index == 2:
            self.available = True
        self.current_index %= 3

    def get_head_dir(self):
        # Calculate the direction vector from last head point to the current head point
        return [
            self.head[self.current_index][0] - self.head[self.last_index][0],
            self.head[self.current_index][1] - self.head[self.last_index][1]
        ]

    def get_tail_dir(self):
        # Calculate the direction vector from last tail point to the current tail point
        return [
            self.tail[self.current_index][0] - self.tail[self.last_index][0],
            self.tail[self.current_index][1] - self.tail[self.last_index][1]
        ]

    def save_to_file(self, cache_dir, pic_num_str):
        file_path = f"{cache_dir}{pic_num_str}.bin"  # Assuming .bin as binary format
        head_dir = self.get_head_dir()
        tail_dir = self.get_tail_dir()

        # Open file and write binary data
        with open(file_path, 'wb') as file:
            file.write(head_dir.tobytes())
            file.write(tail_dir.tobytes())
