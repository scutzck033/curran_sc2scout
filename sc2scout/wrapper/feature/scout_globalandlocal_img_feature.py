import numpy as np
from gym.spaces import Box
import math
from sc2scout.wrapper.feature.img_feat_extractor import ImgFeatExtractor
import sc2scout.envs.scout_macro as sm
from sc2scout.wrapper.util.tmp_target_for_explore import TempTarget

ENEMY_BASE_RANGE = 12


class ScoutlImgFeature(ImgFeatExtractor):
    def __init__(self, compress_width, scout_range, channel_num, reverse=False):
        super(ScoutlImgFeature, self).__init__(compress_width, channel_num, reverse)
        self.global_base_width = math.floor(compress_width / 12)
        self.global_scout_width = math.floor(compress_width / 24)
        self.local_radius = float(scout_range) / 2
        self.local_per_unit = float(scout_range) / compress_width
        self._enemy_base_range = ENEMY_BASE_RANGE
        self.mat_element_value = {'scout': 1.0,
                                  'enemy': 2.0,
                                  'owner_base': 3.0,
                                  'enemy_base': 4.0,
                                  'resource': 5.0,
                                  'local_temp_target': 6.0
                                  }

    def reset(self, env):
        super(ScoutlImgFeature, self).reset(env)
        e_x, e_y = env.unwrapped.enemy_base()
        self.local_tmp_target = TempTarget(ENEMY_BASE_RANGE, e_x, e_y)
        print('ScoutlImgFeature reset')
        print('global_raidus=({},{}), local_raduis=({},{})'.format(self._x_radius, self._y_radius,
                                                                   self.local_radius, self.local_radius))
        print('global per_unit=({},{}), local per_unit=({},{})'.format(self._x_per_unit, self._y_per_unit,
                                                                       self.local_per_unit, self.local_per_unit))

    def extract(self, env, obs, walkaround, back):
        owners, neutrals, enemys = self.unit_dispatch(obs)
        image = np.zeros([self._compress_width, self._compress_width, self._channel_num])

        if not back:
            if not walkaround:  # forward phase
                self.set_pos_global_channel1(env, image, 0, back)  # focus on the minimap for round trip
                self.set_pos_global_channel2(env, image, 1, enemys, back)
                self.set_reverseIndicator_channel(env, image, 2, back)
            else:  # walkaround phase
                self.set_pos_local_channel1(env, image, 0,
                                            enemys)  # focus on the screen for evading enemy and viewing enemy resources
                self.set_pos_local_channel2(env, image, 1, enemys, neutrals)
                self.set_pos_local_channel3(env, image, 2)
        else:  # backward phase
            self.set_pos_global_channel1(env, image, 0, back)  # focus on the minimap for round trip
            self.set_pos_global_channel2(env, image, 1, enemys, back)
            self.set_reverseIndicator_channel(env, image, 2, back)
        # self.set_pos_channel2(env, image, 1, enemys, back)  # set channel two
        # self.set_reverseIndicator_channel(env,image,2,back) #set channel three
        return image

    def obs_space(self):
        low = -np.ones([self._compress_width, self._compress_width, self._channel_num])
        high = np.ones([self._compress_width, self._compress_width, self._channel_num])
        return Box(low, high)

    def unit_dispatch(self, obs):
        units = obs.observation['units']
        owners = []
        neutrals = []
        enemys = []
        for u in units:
            if u.int_attr.alliance == sm.AllianceType.SELF.value:
                owners.append(u)
            elif u.int_attr.alliance == sm.AllianceType.NEUTRAL.value:
                neutrals.append(u)
            elif u.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemys.append(u)
            else:
                continue

        return owners, neutrals, enemys

    # to capture the relative location information between scout and two basees
    def set_pos_global_channel1(self, env, image, channel_id, back_indicator):

        # set owner pos
        owner_base = env.unwrapped.owner_base()
        owner_base_pos = self.pos_2_2d(owner_base[0], owner_base[1])
        self.enhanceRange(image, channel_id, owner_base_pos, self.global_base_width,
                          self.mat_element_value['owner_base'] / len(self.mat_element_value))
        # for u in owners:
        #     i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
        #     if u.unit_type in sm.BUILDING_UNITS:
        #         image[i, j, channel_id] = 1
        #     elif u.unit_type in sm.COMBAT_AIR_UNITS:
        #         image[i, j, channel_id] = 1
        #     elif u.unit_type in sm.COMBAT_UNITS:
        #         image[i, j, channel_id] = 1
        #     else:
        #         image[i, j, channel_id] = 1

        # set enemy pos
        enemy_base = env.unwrapped.enemy_base()
        enemy_base_pos = self.pos_2_2d(enemy_base[0], enemy_base[1])
        self.enhanceRange(image, channel_id, enemy_base_pos, self.global_base_width,
                          self.mat_element_value['enemy_base'] / len(self.mat_element_value))
        # for u in enemys:
        #     i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
        #     if u.unit_type in sm.BUILDING_UNITS:
        #         image[i, j, channel_id] = 1
        #     elif u.unit_type in sm.COMBAT_AIR_UNITS:
        #         continue
        #     elif u.unit_type in sm.COMBAT_UNITS:
        #         continue
        #     else:
        #         image[i, j, channel_id] = 1




        # set scout pos
        scout = env.unwrapped.scout()
        scout_pos = self.pos_2_2d(scout.float_attr.pos_x, scout.float_attr.pos_y)
        self.enhanceRange(image, channel_id, scout_pos, self.global_scout_width,
                          self.mat_element_value['scout'] / len(self.mat_element_value))
        # image[scout_pos[0], scout_pos[1], channel_id] = 2


        if back_indicator:
            image[:, :, channel_id] = (-1) * image[:, :, channel_id]

    # to capture the relative location information between scout and enemies in a rough picture
    def set_pos_global_channel2(self, env, image, channel_id, enemys, back_indicator):
        # set scout pos
        scout = env.unwrapped.scout()
        scout_pos = self.pos_2_2d(scout.float_attr.pos_x, scout.float_attr.pos_y)
        self.enhanceRange(image, channel_id, scout_pos, self.global_scout_width,
                          self.mat_element_value['scout'] / len(self.mat_element_value))

        # set enemies pos
        for u in enemys:
            u_pos = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
            if u.unit_type in sm.COMBAT_AIR_UNITS:
                self.enhanceRange(image, channel_id, u_pos, self.global_scout_width,
                                  self.mat_element_value['enemy'] / len(self.mat_element_value))
            elif u.unit_type in sm.COMBAT_UNITS:
                self.enhanceRange(image, channel_id, u_pos, self.global_scout_width,
                                  self.mat_element_value['enemy'] / len(self.mat_element_value))

        if back_indicator:
            image[:, :, channel_id] = (-1) * image[:, :, channel_id]

    # to capture the relative location information between scout and enemies in a precise picture
    def set_pos_local_channel1(self, env, image, channel_id, enemys):
        cx, cy = self.center_pos(env)
        # set scout pos
        scout = env.unwrapped.scout()
        scout_pos = self.pos_2_2d_local(scout.float_attr.pos_x, scout.float_attr.pos_y, cx, cy)
        self.enhanceRange(image, channel_id, scout_pos, self.global_scout_width,
                          self.mat_element_value['scout'] / len(self.mat_element_value))

        # set enemies pos
        for u in enemys:
            if self.check_in_range(u.float_attr.pos_x, u.float_attr.pos_y,
                                   scout.float_attr.pos_x, scout.float_attr.pos_y, self.local_radius):
                u_pos = self.pos_2_2d_local(u.float_attr.pos_x, u.float_attr.pos_y, cx, cy)
                if u.unit_type in sm.COMBAT_AIR_UNITS:
                    self.enhanceRange(image, channel_id, u_pos, self.global_scout_width,
                                      self.mat_element_value['enemy'] / len(self.mat_element_value))
                elif u.unit_type in sm.COMBAT_UNITS:
                    self.enhanceRange(image, channel_id, u_pos, self.global_scout_width,
                                      self.mat_element_value['enemy'] / len(self.mat_element_value))

    # to capture the relative location information between scout and enemy resources
    def set_pos_local_channel2(self, env, image, channel_id, enemys, neutrals):
        cx, cy = self.center_pos(env)
        # set scout pos
        scout = env.unwrapped.scout()
        enemy_bsae = env.unwrapped.enemy_base()
        scout_pos = self.pos_2_2d_local(scout.float_attr.pos_x, scout.float_attr.pos_y, cx, cy)
        self.enhanceRange(image, channel_id, scout_pos, self.global_scout_width,
                          self.mat_element_value['scout'] / len(self.mat_element_value))

        # set enemy base pos
        for u in enemys:
            if u.unit_type in sm.BASE_UNITS:
                if self.check_in_range(u.float_attr.pos_x, u.float_attr.pos_y, cx, cy, self.local_radius):
                    u_pos = self.pos_2_2d_local(u.float_attr.pos_x, u.float_attr.pos_y, cx, cy)
                    self.enhanceRange(image, channel_id, u_pos, self.global_scout_width,
                                      self.mat_element_value['enemy_base'] / len(self.mat_element_value))
        # set enemy resources pos
        for u in neutrals:
            if self.check_in_range(u.float_attr.pos_x, u.float_attr.pos_y, cx, cy, self.local_radius) \
                    and self.check_in_range(u.float_attr.pos_x, u.float_attr.pos_y, enemy_bsae[0], enemy_bsae[1], 10):
                if u.unit_type in sm.MINERAL_UNITS or u.unit_type in sm.VESPENE_UNITS:
                    u_pos = self.pos_2_2d_local(u.float_attr.pos_x, u.float_attr.pos_y, cx, cy)
                    self.enhanceRange(image, channel_id, u_pos, self.global_scout_width,
                                      self.mat_element_value['resource'] / len(self.mat_element_value))

    # to capture the relative location information between scout and temp target while at exploring state
    def set_pos_local_channel3(self, env, image, channel_id):
        cx, cy = self.center_pos(env)
        # set scout pos
        scout_pos = self.pos_2_2d(cx, cy)
        self.enhanceRange(image, channel_id, scout_pos, self.global_scout_width,
                          self.mat_element_value['scout'] / len(self.mat_element_value))

        # set temp target pos
        if self.local_tmp_target.curr_target_index() is None:
            self.local_tmp_target.setCurrTarget(cx, cy)

        curr_target_pos = self.local_tmp_target.curr_target_pos()

        u_pos = self.pos_2_2d(curr_target_pos[0], curr_target_pos[1])
        self.enhanceRange(image, channel_id, u_pos, self.global_base_width,
                          self.mat_element_value['local_temp_target'] / len(self.mat_element_value))

        curr_dist = env.unwrapped._calculate_distances(cx, cy,
                                                       curr_target_pos[0],
                                                       curr_target_pos[1])
        if curr_dist < 2:
            self.local_tmp_target.set_curr_target_index()

        # set owner pos
        owner_base = env.unwrapped.owner_base()
        owner_base_pos = self.pos_2_2d(owner_base[0], owner_base[1])
        self.enhanceRange(image, channel_id, owner_base_pos, self.global_base_width,
                          self.mat_element_value['owner_base'] / len(self.mat_element_value))

    def set_reverseIndicator_channel(self, env, image, channel_id, back_indicator):
        # if target is the enemy base
        if not back_indicator:
            image[:, :, channel_id] = 0.1
        # if target is the home base
        else:
            image[:, :, channel_id] = -0.1

    def pos_2_2d_local(self, pos_x, pos_y, cx, cy):
        pos_x = (pos_x - cx) + self.local_radius
        pos_y = (pos_y - cy) + self.local_radius
        i = math.floor(pos_x / self.local_per_unit)
        j = math.floor(pos_y / self.local_per_unit)
        return i, j

    def check_in_range(self, pos_x, pos_y, unit_x, unit_y, radius):

        x_low = unit_x - radius
        x_high = unit_x + radius
        y_low = unit_y - radius
        y_high = unit_y + radius
        if pos_x > x_high or pos_x < x_low:
            return False
        if pos_y > y_high or pos_y < y_low:
            return False
        return True

    def center_pos(self, env):
        scout = env.unwrapped.scout()
        cx = scout.float_attr.pos_x
        cy = scout.float_attr.pos_y
        return cx, cy

    def enhanceRange(self, image, channel_id, pos, enhance_width, value=1):
        for i in range(max(0, pos[0] - enhance_width), min(self._compress_width, pos[0] + enhance_width)):
            for j in range(max(0, pos[1] - enhance_width), min(self._compress_width, pos[1] + enhance_width)):
                image[i, j, channel_id] += value

