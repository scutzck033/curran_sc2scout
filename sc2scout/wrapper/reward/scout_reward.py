from sc2scout.envs import scout_macro as sm
from sc2scout.wrapper.reward.reward import Reward
from sc2scout.wrapper.util.dest_range import DestRange
import numpy as np
from sc2scout.wrapper.util.tmp_target_for_explore import TempTarget

ARRIVED_DIST_GAP = 5

class HomeReward(Reward):
    def __init__(self, weight,back=False, negative=False):
        super(HomeReward, self).__init__(weight)
        self._last_dist = None
        self._back = back
        self._negative = negative

    def reset(self, obs, env):
        self._last_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, done, env):
        tmp_dist = self._compute_dist(env)
        if self._back:
            if self._negative:
                if tmp_dist >= self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
                if tmp_dist > self._last_dist:
                    self.rwd = self.w * -1
                elif tmp_dist < self._last_dist:
                    self.rwd = self.w * 1
                else:
                    if tmp_dist > 0:
                        self.rwd = self.w * -1
                    else:
                        self.rwd = 0
        else:
            if self._negative:
                if tmp_dist <= self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
                if tmp_dist > self._last_dist:
                    self.rwd = self.w * 1
                elif tmp_dist < self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
        print('home_rwd=', self.rwd)
        self._last_dist = tmp_dist

    def _compute_dist(self, env):
        scout = env.scout()
        home = env.owner_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     home[0], home[1])
        return dist

class HomeArrivedReward(Reward):
    def __init__(self, weight=10):
        super(HomeArrivedReward, self).__init__(weight)
        self._once = False

    def reset(self, obs, env):
        self._once = False

    def compute_rwd(self, obs, reward, done, env):
        if self._once:
            self.rwd = 0
            return
        tmp_dist = self._compute_dist(env)
        if tmp_dist <= ARRIVED_DIST_GAP:
            self.rwd = 1 * self.w
            self._once = True
        else:
            self.rwd = 0
        print('home_arrived reward=', self.rwd)

    def _compute_dist(self, env):
        scout = env.scout()
        home = env.owner_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     home[0], home[1])
        return dist

class EnemyBaseReward(Reward):
    def __init__(self, weight,back=False, negative=False):
        super(EnemyBaseReward, self).__init__(weight)
        self._last_dist = None
        self._back = back
        self._negative = negative

    def reset(self, obs, env):
        self._last_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, done, env):
        tmp_dist = self._compute_dist(env)
        if self._back:
            if self._negative:
                if tmp_dist <= self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
                if tmp_dist < self._last_dist:
                    self.rwd = self.w * -1
                elif tmp_dist > self._last_dist:
                    self.rwd = self.w * 1
                else:
                    self.rwd = 0
        else:
            if self._negative:
                if tmp_dist >= self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
                if tmp_dist < self._last_dist:
                    self.rwd = self.w * 1
                elif tmp_dist > self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
        self._last_dist = tmp_dist
        print('enemy_base_rwd=', self.rwd)

    def _compute_dist(self, env):
        scout = env.scout()
        enemy_base = env.enemy_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     enemy_base[0], enemy_base[1])
        return dist

class EnemyBaseArrivedReward(Reward):
    def __init__(self, weight=10):
        super(EnemyBaseArrivedReward, self).__init__(weight)
        self._once = False

    def reset(self, obs, env):
        self._once = False

    def compute_rwd(self, obs, reward, done, env):
        if self._once:
            self.rwd = 0
            return
        tmp_dist = self._compute_dist(env)
        if tmp_dist <= ARRIVED_DIST_GAP:
            self.rwd = 1 * self.w
            self._once = True
        else:
            self.rwd = 0
        print('enemy_base_arrived reward=', self.rwd)

    def _compute_dist(self, env):
        scout = env.scout()
        enemy_base = env.enemy_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     enemy_base[0], enemy_base[1])
        return dist

class ViewEnemyReward(Reward):
    def __init__(self, weight=3):
        super(ViewEnemyReward, self).__init__(weight)
        self._unit_set = set([])
        ''' the tag of enemybase will be repeatly changed while scout is nearby'''
        self._enemy_base_once = False

    def reset(self, obs, env):
        self._unit_set = set([])
        self._enemy_base_once = False

    def compute_rwd(self, obs, reward, done, env):
        enemy_units = []
        find_base = False
        units = obs.observation['units']
        for unit in units:
            if unit.int_attr.alliance == sm.AllianceType.ENEMY.value:
                if unit.unit_type in sm.BASE_UNITS:
                    if not self._enemy_base_once:
                        self._enemy_base_once = True
                        find_base = True
                else:
                    enemy_units.append(unit.tag)

        count = 0
        for eu in enemy_units:
            if eu in self._unit_set:
                pass
            else:
                count += 1
                self._unit_set.add(eu)
        if find_base:
            count += 1

        self.rwd = count * self.w
        print('view enemy count=', count, ';reward=', self.rwd)

MIN_DIST_ERROR = 2.0

class MinDistReward(Reward):
    def __init__(self, negative=False, weight=1):
        super(MinDistReward, self).__init__(weight)
        self._min_dist = None
        self._negative = negative

    def reset(self, obs, env):
        self._min_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, done, env):
        tmp_dist = self._compute_dist(env)
        if self._negative:
            if tmp_dist >= self._min_dist + MIN_DIST_ERROR:
                self.rwd = self.w * -1
            else:
                self.rwd = 0
        else:
            if tmp_dist > self._min_dist + MIN_DIST_ERROR:
                self.rwd = self.w * -1
            else:
                self.rwd = self.w * 1

        if self._min_dist > tmp_dist:
            self._min_dist = tmp_dist
        print("MinDistReward=",self.rwd)

    def _compute_dist(self, env):
        scout = env.scout()
        home = env.owner_base()
        enemy_base = env.enemy_base()
        home_dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     home[0], home[1])

        enemy_dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     enemy_base[0], enemy_base[1])
        return (home_dist + enemy_dist)

class OnewayFinalReward(Reward):
    def __init__(self, weight=50):
        super(OnewayFinalReward, self).__init__(weight)

    def reset(self, obs, env):
        self._dest = DestRange(env.enemy_base())

    def compute_rwd(self, obs, reward, done, env):
        self._compute_rwd(env)
        if done:
            if self._dest.hit:
                #print('compute final rwd, hit rwd=', self.w * 2)
                self.rwd = self.w * 2
            elif self._dest.enter:
                #print('compute final rwd, enter rwd=', self.w * 1)
                self.rwd = self.w * 1
            else:
                self.rwd = self.w * -1
        else:
            self.rwd = 0

    def _compute_rwd(self, env):
        scout = env.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)

        self._dest.check_enter(pos)
        self._dest.check_hit(pos)
        self._dest.check_leave(pos)

class RoundTripFinalReward(Reward):
    def __init__(self, weight=50):
        super(RoundTripFinalReward, self).__init__(weight)
        self._back = False

    def reset(self, obs, env):
        self._dest = DestRange(env.enemy_base())
        self._src = DestRange(env.owner_base())

    def compute_rwd(self, obs, reward, done, env):
        scout = env.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        self._check(pos)
        if done:
            if self._dest.hit and self._src.hit:
                self.rwd = self.w * 2
            elif self._dest.enter and self._src.enter:
                self.rwd = self.w * 1
            else:
                self.rwd = 0
        else:
            self.rwd = 0
        print("RoundTripFinalReward",self.rwd)

    def _check(self, pos):
        if not self._back:
            self._check_dest(pos)
        else:
            self._check_src(pos)
        if self._dest.enter and self._dest.leave:
            if not self._back:
                self._back = True

    def _check_dest(self, pos):
        self._dest.check_enter(pos)
        self._dest.check_hit(pos)
        self._dest.check_leave(pos)

    def _check_src(self, pos):
        self._src.check_enter(pos)
        self._src.check_hit(pos)
        self._src.check_leave(pos)

class HitEnemyBaseReward(Reward):
    def __init__(self, weight=10):
        super(HitEnemyBaseReward, self).__init__(weight)

    def reset(self, obs, env):
        self._dest = DestRange(env.enemy_base())
        self._hit_once = False

    def compute_rwd(self, obs, reward, done, env):
        scout = env.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        self._dest.check_hit(pos)
        self._dest.check_enter(pos)
        self._dest.check_leave(pos)

        if not self._hit_once and (self._dest.hit or (self._dest.enter and self._dest.leave)):
            self.rwd = self.w * 1
            self._hit_once = True
        else:
            self.rwd = 0
        print("HitEnemyBaseReward",self.rwd,self._hit_once)

SCOUT_RANGE_WIDTH = 2 # set it even for simple
STEP_CHECKING_LEN = 24 # compute_rw activated every 40 steps

class AreaOfOverlapReward(Reward):
    def __init__(self,weight=1):
        super(AreaOfOverlapReward, self).__init__(weight)
        self._range_width = SCOUT_RANGE_WIDTH
        self._step_checking_len = STEP_CHECKING_LEN

    def reset(self, obs, env):
        scout = env.unwrapped.scout()
        pos = (int(scout.float_attr.pos_x), int(scout.float_attr.pos_y))
        self._last_scout_range_map = self.createScoutRangeMap(pos, env)
        self._last_overlap_percent = 0

    def compute_rwd(self, obs, reward, done, env):
        curr_step = env.unwrapped.curr_step()
        if curr_step % self._step_checking_len == 0:
            scout = env.scout()
            pos = (int(scout.float_attr.pos_x), int(scout.float_attr.pos_y))
            curr_scoutRangeMap = self.createScoutRangeMap(pos,env)
            area = self.computeAreaOfOverlap(curr_scoutRangeMap,self._last_scout_range_map)
            curr_overlap_percent = float(area)/((self._range_width+1)**2)
            if curr_overlap_percent >= self._last_overlap_percent:
                self.rwd = -curr_overlap_percent * self.w
            else:
                self.rwd = 5*(self._last_overlap_percent-curr_overlap_percent) * self.w
            if curr_overlap_percent == 0:
                self.rwd = 3
            self._last_overlap_percent = curr_overlap_percent
            self._last_scout_range_map = curr_scoutRangeMap
            print("AreaOfOverlapReward",self.rwd)


    def createScoutRangeMap(self,pos,env):
        map_size = env.unwrapped.map_size()
        range_map = np.zeros(shape=(map_size[0], map_size[1]))
        for i in range(int(pos[0]-self._range_width/2),int(pos[0]+self._range_width/2+1)):
            for j in range(int(pos[1]-self._range_width/2),int(pos[1]+self._range_width/2+1)):
                range_map[i][j]=1
        return range_map

    def computeAreaOfOverlap(self,map1,map2):
        area = 0
        for i in range(map1.shape[0]):
            for j in range(map1.shape[1]):
                if map1[i][j] and map2[i][j]:
                    area+=1
        return area

class ViewEnemyResourcesAndBase(Reward):
    def __init__(self, weight=3):
        super(ViewEnemyResourcesAndBase, self).__init__(weight)
        self._unit_set = set([])
        ''' the tag of enemybase will be repeatly changed while scout is nearby'''
        self._enemy_base_once = False

    def reset(self, obs, env):
        self._unit_set = set([])
        self._enemy_base_once = False

    def compute_rwd(self, obs, reward, done, env):
        enemy_base = env.unwrapped.enemy_base()
        scout = env.unwrapped.scout()
        enemy_units = []
        find_base = False
        units = obs.observation['units']

        # for unit in units:
        #     if unit.int_attr.alliance == sm.AllianceType.ENEMY.value:
        #         if unit.unit_type in sm.BASE_UNITS:
        #             enemy_base = unit
        #
        for u in units:
            #judge the enemy resources, no need the precise enemy base location
            if u.int_attr.alliance == sm.AllianceType.NEUTRAL.value \
                    and env.unwrapped._calculate_distances(u.float_attr.pos_x,u.float_attr.pos_y,enemy_base[0],enemy_base[1])<10:
                    enemy_units.append((u.float_attr.pos_x,u.float_attr.pos_y))
            #precise one, agree with the obervation
            elif u.int_attr.alliance == sm.AllianceType.ENEMY.value and u.unit_type in sm.BASE_UNITS:
                if (not self._enemy_base_once) and (env.unwrapped._unit_dist(scout,u)<=8):
                    self._enemy_base_once = True
                    find_base = True

        count = 0
        for eu in enemy_units:
            if eu in self._unit_set:
                pass
            else:
                if env.unwrapped._calculate_distances(eu[0],eu[1],scout.float_attr.pos_x,scout.float_attr.pos_y)<=8:
                    count += 1
                    self._unit_set.add(eu)
        if find_base:
            count += 0

        self.rwd = count * self.w
        print('view enemy resouces count=', count, ';reward=', self.rwd,';find base',self._enemy_base_once)

class ExploreStateRwd(Reward):
    def __init__(self,weight=1):
        super(ExploreStateRwd,self).__init__(weight)

    def reset(self, obs, env):
        self.reward_once = False

    def compute_rwd(self, obs, reward, done, env):
        if not self.reward_once:
            self.rwd = 1 * self.w
            self.reward_once = True
        else:
            self.rwd = 0
        print("ExploreStateRwd",self.rwd)


class BackwardStateRwd(Reward):
    def __init__(self, weight=1):
        super(BackwardStateRwd, self).__init__(weight)

    def reset(self, obs, env):
        self.reward_once = False

    def compute_rwd(self, obs, reward, done, env):
        if not self.reward_once:
            self.rwd = 1 * self.w
            self.reward_once = True
        else:
            self.rwd = 0
        print("BackwardStateRwd",self.rwd)

ENEMY_BASE_RANGE = 12

class ExploreAcclerateRwd(Reward):

    def __init__(self, weight=1):
        super(ExploreAcclerateRwd, self).__init__(weight)
        self._last_dist_to_Home = None

    def reset(self, obs, env):
        scout = env.scout()
        self._last_health = scout.float_attr.health
        e_x, e_y = env.unwrapped.enemy_base()
        self.tmp_target = TempTarget(ENEMY_BASE_RANGE, e_x, e_y)
        self.target_count = 0
        self._last_dist_to_Home = self._compute_dist(env)

    def compute_rwd(self, obs, reward, done, env):
        scout_x, scout_y = env.unwrapped.scout().float_attr.pos_x, env.unwrapped.scout().float_attr.pos_y
        if self.tmp_target.curr_target_index() is None:
            self.tmp_target.setCurrTarget(scout_x, scout_y)


        curr_target_pos = self.tmp_target.curr_target_pos()
        curr_dist = env.unwrapped._calculate_distances(scout_x, scout_y,
                                            curr_target_pos[0],
                                            curr_target_pos[1])
        tmp_dist_to_home = self._compute_dist(env)
        
        curr_health = env.unwrapped.scout().float_attr.health
        
        if curr_dist < self.tmp_target.last_target_dist():
            if curr_health < self._last_health:
                self.rwd = -3 * self.w
            else:
                self.rwd = 0
        else:
            if curr_health < self._last_health:
                self.rwd = -3 * self.w
            else:
                if tmp_dist_to_home <= self._last_dist_to_Home:
                    self.rwd = -2 * self.w
                else:
                    self.rwd = -1 * self.w
                
        print("ExploreAcclerateRwd", self.rwd, "curr_dist", curr_dist, "last_dist", self.tmp_target.last_target_dist(),
              "reached target", self.target_count)

        self.tmp_target.set_last_target_dist(curr_dist)

        if curr_dist < 4:
            self.tmp_target.set_curr_target_index()
            next_target_pos = self.tmp_target.curr_target_pos()
            curr_dist = env.unwrapped._calculate_distances(scout_x, scout_y,
                                            next_target_pos[0],
                                            next_target_pos[1])
            self.tmp_target.set_last_target_dist(curr_dist)
            self.target_count = self.target_count+1

        self._last_dist_to_Home = tmp_dist_to_home
        self._last_health = curr_health

        

    def _compute_dist(self, env):
        scout = env.scout()
        home = env.owner_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     home[0], home[1])
        return dist
