import numpy as np
from sc2scout.wrapper.util.dest_range import DestRange
from sc2scout.wrapper.explore_enemy.auxiliary_wrapper import RoundTripTerminalWrapper

JUDGE_WALKAROUND_DISTANCE = 12
ENEMY_BASE_RANGE = 12
SCOUT_RANGE = 8
EXPLORE_STEP = 800

class ExploreWithEvadeTerminalWrapper(RoundTripTerminalWrapper):
    def __init__(self, env):
        super(ExploreWithEvadeTerminalWrapper, self).__init__(env)
        self._judge_walkaround_dist = JUDGE_WALKAROUND_DISTANCE
        self._enemy_base_range_width = ENEMY_BASE_RANGE
        self._scout_range_width = SCOUT_RANGE
        self._map_size = self.env.unwrapped.map_size()
        self._explore_step_required = EXPLORE_STEP

    def _reset(self):
        obs = self.env._reset()
        self._judge_walkaround = False
        self._task_finished = False
        self._judge_back = False
        self._enemy_base_range_map = self.createRangeMap(self.env.unwrapped.enemy_base(),self._enemy_base_range_width)
        self._home_base = DestRange(self.env.unwrapped.owner_base())
        self._curr_explore_step = 0

        return obs

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        done = self._check_terminal(obs,done)
        self.judge_walkAround()
        self.judge_back()
        info = {'walkaround':self._judge_walkaround,'back':self._judge_back,'finished':self._task_finished}
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if done:
            return done

        scout = self.env.unwrapped.scout()
        # scout_health = scout.float_attr.health
        # max_health = scout.float_attr.health_max

        # print("self._home_base.enter",self._home_base.enter)

        if self._judge_back:
            self._home_base.check_enter((scout.float_attr.pos_x, scout.float_attr.pos_y))
            if self._home_base.enter:
                self._task_finished = True
                return True

        survive = self.env.unwrapped.scout_survive()
        if survive:
            return done
        else:
            return True

    def createRangeMap(self,pos,range_width):

        range_map = np.zeros(shape=(self._map_size[0], self._map_size[1]))
        for i in range(int(pos[0]-range_width),int(pos[0]+range_width+1)):
            for j in range(int(pos[1]-range_width),int(pos[1]+range_width+1)):
                range_map[i][j]=1
        return range_map

    def updateEnemyBaseMap(self,scout_map):
        for i in range(self._enemy_base_range_map.shape[0]):
            for j in range(self._enemy_base_range_map.shape[1]):
                if self._enemy_base_range_map[i][j] and scout_map[i][j]:
                    self._enemy_base_range_map[i][j] = 0

    def check_enemybase_in_range(self):
        pos_x,pos_y = self.env.unwrapped.enemy_base()[0],self.env.unwrapped.enemy_base()[1]
        scout = self.env.unwrapped.scout()
        x_low = scout.float_attr.pos_x - self._judge_walkaround_dist
        x_high = scout.float_attr.pos_x + self._judge_walkaround_dist
        y_low = scout.float_attr.pos_y - self._judge_walkaround_dist
        y_high = scout.float_attr.pos_y + self._judge_walkaround_dist
        if pos_x > x_high or pos_x < x_low:
            return False
        if pos_y > y_high or pos_y < y_low:
            return False
        return True

    def judge_walkAround(self):
        if (not self._judge_walkaround) and (self.check_enemybase_in_range()):
            self._judge_walkaround = True


    def judge_back(self):
        print("curr_explore_step:",self._curr_explore_step)
        if self._judge_walkaround:
            self._curr_explore_step += 1
            if (not self._judge_back) and (self._curr_explore_step > self._explore_step_required):
                self._judge_back = True
#        scout = self.env.unwrapped.scout()
#        curr_scout_map=self.createRangeMap([scout.float_attr.pos_x,scout.float_attr.pos_y],self._scout_range_width)
#        self.updateEnemyBaseMap(curr_scout_map)
#        if (1 in self._enemy_base_range_map):
#            pass
#        else:
#            self._judge_back = True
