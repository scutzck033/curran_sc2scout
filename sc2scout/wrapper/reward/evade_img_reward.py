import math

from sc2scout.wrapper.reward.reward import Reward
from sc2scout.envs import scout_macro as sm

class EvadeUnderAttackRwd(Reward):
    def __init__(self, weight=1):
        super(EvadeUnderAttackRwd, self).__init__(weight)

    def reset(self, obs, env):
        scout = env.scout()
        self._last_health = scout.float_attr.health
        self._max_health = scout.float_attr.health_max
        self._rwd_sum = 0.0
        self._max_rwd_sum = 100.0
        self._last_left_rwd = 100

    def compute_rwd(self, obs, reward, done, env):
        curr_health = env.unwrapped.scout().float_attr.health
        survive = env.unwrapped.scout_survive()
        if curr_health == self._last_health:
            self.rwd = 0
            return

        rwd = self._compute_rwd(curr_health)
        self.rwd = self.w * rwd
        self._last_health = curr_health
        print('underattack rwd; rwd={}, rwd_sum={}'.format(self.rwd, self._rwd_sum))

    def _compute_rwd(self, curr_health):
        left_rwd = math.floor(int((curr_health / self._max_health) * self._max_rwd_sum))
        rwd = left_rwd - self._last_left_rwd
        self._last_left_rwd = left_rwd
        self._rwd_sum += rwd
        return rwd 

class EvadeFinalRwd(Reward):
    def __init__(self, weight=100):
        super(EvadeFinalRwd, self).__init__(weight)

    def reset(self, obs, env):
        pass

    def compute_rwd(self, obs, reward, done, env):
        if done:
            survive = env.unwrapped.scout_survive()
            if survive:
                self.rwd = 1 * self.w
            else:
                self.rwd = -1 * self.w
        else:
            self.rwd = 0
        print('evade final_rwd=', self.rwd)

class EvadeInTargetRangeRwd(Reward):
    def __init__(self, weight=1):
        super(EvadeInTargetRangeRwd, self).__init__(weight)
        self._range = 30
        self._target = None
        self._compress_width = 32
        self._range_width = 12
 
    def reset(self, obs, env):
        target = env.unwrapped.enemy_base()
        map_size = env.unwrapped.map_size()
        x_per_unit = map_size[0] / self._compress_width
        y_per_unit = map_size[1] / self._compress_width
        x_radius = (x_per_unit * self._range_width) / 2
        y_radius = (y_per_unit * self._range_width) / 2
        self._x_low = target[0] - x_radius
        self._x_high = target[0] + x_radius
        self._y_low = target[1] - y_radius
        self._y_high = target[1] + y_radius
        print('Evade per_unit=({},{}), radius=({},{}), x_range=({},{}), y_range=({},{}), target={}'.format(
            x_per_unit, y_per_unit, x_radius, y_radius,
            self._x_low, self._x_high, self._y_low, self._y_high, target))

    def compute_rwd(self, obs, reward, done, env):
        scout = env.unwrapped.scout()
        in_range = self._in_range(env)
        if in_range:
            self.rwd = 0
        else:
            self.rwd = -1 * self.w
        #print('InRange scout=({}, {}), Rwd={}'.format(
        #    scout.float_attr.pos_x, scout.float_attr.pos_y, self.rwd))

    def _in_range(self, env):
        scout = env.unwrapped.scout()
        if scout.float_attr.pos_x > self._x_high:
            return False
        elif scout.float_attr.pos_x < self._x_low:
            return False
        elif scout.float_attr.pos_y > self._y_high:
            return False
        elif scout.float_attr.pos_y < self._y_low:
            return False
        else:
            return True

RANGE_DETECTION_OF_SCOUT = 10

class EnemyInRangeRwd(Reward):
    def __init__(self, weight=1):
        super(EnemyInRangeRwd, self).__init__(weight)
        self._range_width = RANGE_DETECTION_OF_SCOUT

    def reset(self, obs, env):
        pass

    def compute_rwd(self, obs, reward, done, env):
        self.rwd = 0

        scout = env.unwrapped.scout()
        units = obs.observation['units']
        for u in units:
            if env.unwrapped._check_enemy(u):
                # if scout is to closed to enemies
                if env.unwrapped._unit_dist(scout,u) < self._range_width:
                    self.rwd = -1 * self.w
                else:
                    self.rwd = 0
        print("EnemyInRangeRwd=",self.rwd)


class EvadeDistanceReward(Reward):
    def __init__(self, weight=1):
        super(EvadeDistanceReward, self).__init__(weight)

    def reset(self, obs, env):
        self._map_size = env.map_size()
        self.enemies_last_dist = {}

    def compute_rwd(self, obs, reward, done, env):
        scout = env.unwrapped.scout()
        units = obs.observation['units']
        for u in units:
            if env.unwrapped._check_enemy(u):
                scout_x = scout.float_attr.pos_x / float(self._map_size[0])
                scout_y = scout.float_attr.pos_y / float(self._map_size[1])
                enemy_x = u.float_attr.pos_x / float(self._map_size[0])
                enemy_y = u.float_attr.pos_y / float(self._map_size[1])
                curr_dist = env._calculate_distances(scout_x, scout_y, enemy_x, enemy_y)
                if self.enemies_last_dist.get(u.tag):
                    if curr_dist<=self.enemies_last_dist[u.tag]:
                        self.rwd = -1 * self.w
                    else:
                        self.rwd = 0
                else:
                    self.enemies_last_dist[u.tag]=curr_dist

        print('EvadeDistanceReward reward=', self.rwd)