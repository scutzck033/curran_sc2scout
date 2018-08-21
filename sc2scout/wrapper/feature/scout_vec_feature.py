from gym.spaces import Box
from sc2scout.envs import scout_macro as sm
import numpy as np

from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
from sc2scout.wrapper.util.dest_range import DestRange


SCOUT_IN_RANGE = 1
SCOUT_OUT_RANGE = 0

class ScoutSimpleFeature(FeatureExtractor):
    def __init__(self):
        self._reverse = False
        self._map_size = None
        self._env = None

    def reset(self, env):
        self.env = env
        self._map_size = self.env.unwrapped.map_size()
        self._reverse = self._judge_reverse(env)

    def obs_space(self):
        low = np.zeros(6)
        high = np.ones(6)
        return Box(low, high)

    def extract(self, env, obs):
        scout = env.unwrapped.scout()
        scout_raw_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = env.unwrapped.owner_base()
        enemy_pos = env.unwrapped.enemy_base()
        scout_pos = self._pos_transfer(scout_raw_pos[0], scout_raw_pos[1])
        home_pos = self._pos_transfer(home_pos[0], home_pos[1])
        enemy_pos = self._pos_transfer(enemy_pos[0], enemy_pos[1])

        features = []
        features.append(float(scout_pos[0]) / self._map_size[0])
        features.append(float(scout_pos[1]) / self._map_size[1])
        features.append(float(abs(home_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(home_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(float(abs(enemy_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(enemy_pos[1] - scout_pos[1])) / self._map_size[1])

        return features

    def _judge_reverse(self, env):
        scout = self.env.unwrapped.scout()
        if scout.float_attr.pos_x < scout.float_attr.pos_y:
            return False
        else:
            return True

    def _pos_transfer(self, x, y):
        if not self._reverse:
            return (x, y)
        cx = self._map_size[0] / 2
        cy = self._map_size[1] / 2
        pos_x = 0.0
        pos_y = 0.0
        if x > cx:
            pos_x = cx - abs(x - cx)
        else:
            pos_x = cx + abs(x - cx)

        if y > cy:
            pos_y = cy - abs(y - cy)
        else:
            pos_y = cy + abs(y - cy)

        return (pos_x, pos_y)

class ScoutVecFeature(FeatureExtractor):
    def __init__(self):
        super(ScoutVecFeature, self).__init__()
        self._dest = None
        self._src = None
        self._reverse = False
        self._map_size = None
        self.env = None

    def reset(self, env):
        self.env = env
        self._dest = DestRange(env.unwrapped.enemy_base())
        self._src = DestRange(env.unwrapped.owner_base())
        self._map_size = self.env.unwrapped.map_size()
        self._reverse = self._judge_reverse(env)

    def obs_space(self):
        low = np.zeros(8)
        high = np.ones(8)
        return Box(low, high)

    def extract(self, env, obs):
        scout = env.unwrapped.scout()
        scout_raw_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = env.unwrapped.owner_base()
        enemy_pos = env.unwrapped.enemy_base()
        scout_pos = self._pos_transfer(scout_raw_pos[0], scout_raw_pos[1])
        home_pos = self._pos_transfer(home_pos[0], home_pos[1])
        enemy_pos = self._pos_transfer(enemy_pos[0], enemy_pos[1])

        features = []
        features.append(float(scout_pos[0]) / self._map_size[0])
        features.append(float(scout_pos[1]) / self._map_size[1])
        #features.append(float(home_pos[0]) / self._map_size[0])
        #features.append(float(home_pos[1]) / self._map_size[1])
        #features.append(float(enemy_pos[0]) / self._map_size[0])
        #features.append(float(enemy_pos[1]) / self._map_size[1])
        features.append(float(abs(home_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(home_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(float(abs(enemy_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(enemy_pos[1] - scout_pos[1])) / self._map_size[1])

        if self._dest.in_range(scout_raw_pos):
            features.append(float(1))
        else:
            features.append(float(0))

        if self._src.in_range(scout_raw_pos):
            features.append(float(1))
        else:
            features.append(float(0))

        return features

    def _judge_reverse(self, env):
        scout = self.env.unwrapped.scout()
        if scout.float_attr.pos_x < scout.float_attr.pos_y:
            return False
        else:
            return True

    def _pos_transfer(self, x, y):
        if not self._reverse:
            return (x, y)
        cx = self._map_size[0] / 2
        cy = self._map_size[1] / 2
        pos_x = 0.0
        pos_y = 0.0
        if x > cx:
            pos_x = cx - abs(x - cx)
        else:
            pos_x = cx + abs(x - cx)

        if y > cy:
            pos_y = cy - abs(y - cy)
        else:
            pos_y = cy + abs(y - cy)

        return (pos_x, pos_y)

NUM_OF_ENEMIES = 3

class ScoutStaticsticVec(FeatureExtractor):
    def __init__(self):
        super(ScoutStaticsticVec, self).__init__()
        self.env = None

    def reset(self, env):
        print("ScoutStaticsticVec obs reset")

    def obs_space(self):
        low = np.zeros(NUM_OF_ENEMIES)
        high = np.ones(NUM_OF_ENEMIES)
        return Box(low, high)

    def extract(self, env, obs, action):
        scout = env.unwrapped.scout()
        features = []

        curr_enemy_count = self._count_enemies(obs)
        # if scout.bool_attr.is_flying:
        #     features.append(1)  # flying
        # else:
        #     features.append(0)
        features.append(float(action)/8)# feature normilization
        # features.append(scout.float_attr.facing)
        # features.append(scout.float_attr.radius)
        features.append(scout.float_attr.health/scout.float_attr.health_max)# feature normilization
        features.append(float(curr_enemy_count) / NUM_OF_ENEMIES)  # enemy_number feature after normilization

        return np.array(features)

    def _count_enemies(self,obs):
        enemy_count = 0
        units = obs.observation['units']
        for u in units:
            if (u.int_attr.alliance == sm.AllianceType.ENEMY.value) and (
                    u.unit_type in sm.COMBAT_UNITS or u.unit_type in sm.COMBAT_AIR_UNITS):
                enemy_count += 1
        return enemy_count




