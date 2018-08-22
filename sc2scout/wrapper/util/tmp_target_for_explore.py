import numpy as np

class TempTarget():
    def __init__(self,base_range,base_x,base_y):
        self._enemy_base_range = base_range
        self._tmp_target_list = [
            (base_x - self._enemy_base_range, base_y - self._enemy_base_range),
            (base_x - self._enemy_base_range, base_y + self._enemy_base_range),
            (base_x + self._enemy_base_range, base_y - self._enemy_base_range),
            (base_x + self._enemy_base_range, base_y + self._enemy_base_range)
        ]
        self._curr_target_index = None
        self._last_target_dist = None

    def curr_target_index(self):
        return self._curr_target_index

    def last_target_dist(self):
        return self._last_target_dist

    def curr_target_pos(self):
        return self._tmp_target_list[self._curr_target_index][0],self._tmp_target_list[self._curr_target_index][1]

    def set_curr_target_index(self):
        self._curr_target_index = (self._curr_target_index + 1) % (len(self._tmp_target_list))

    def set_last_target_dist(self,dist):
        self._last_target_dist = dist

    def setCurrTarget(self,u_x,u_y):
        tmp_index = 0
        tmp_dist = np.float("inf")
        for i in range(len(self._tmp_target_list)):
            curr_dist = self._calculate_distances(u_x, u_y,
                                            self._tmp_target_list[i][0],
                                            self._tmp_target_list[i][1])
            if curr_dist < tmp_dist:
                tmp_index = i
                tmp_dist = curr_dist
        self._curr_target_index = tmp_index
        self._last_target_dist = tmp_dist

    def _calculate_distances(self, x1, y1, x2, y2):
        x = abs(x1 - x2)
        y = abs(y1 - y2)
        distance = x ** 2 + y ** 2
        return distance ** 0.5



