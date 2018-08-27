import gym
from sc2scout.envs import SC2GymEnv
import numpy as np
from gym.spaces import Box

class ZergScoutWrapper(gym.Wrapper):
    def __init__(self, env):
        print('ZergScoutWrapper initilize')
        super(ZergScoutWrapper, self).__init__(env)
        print('act_space_shape=', self.action_space.shape)
        print('obs_space_shape=', self.observation_space.shape)

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        print("current step",self.env.unwrapped.curr_step())
        return self.env._step(action)

class ZergScoutWrapper2(gym.Wrapper):
    def __init__(self, env):
        print('ZergScoutWrapper2 initilize')
        super(ZergScoutWrapper2, self).__init__(env)
        print('act_space_shape=', self.action_space.shape)
        print('img_obs_space_shape={},vec_obs_space_shape={}'.format(self.observation_space[0].shape,
                                                                     self.observation_space[1].shape))

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        scout = self.env.unwrapped.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        print("current step",self.env.unwrapped.curr_step())
        print("scout pos:({},{})".format(pos[0],pos[1]))
        return self.env._step(action)
