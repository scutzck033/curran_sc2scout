import gym
from gym.spaces import Box
import numpy as np
from sc2scout.wrapper.feature.scout_globalandlocal_img_feature import ScoutlImgFeature
from sc2scout.wrapper.feature.scout_vec_feature import ScoutStaticsticVec

class ExploreWithEvadeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ExploreWithEvadeObsWrapper, self).__init__(env)
        self._obs = (ScoutlImgFeature(compress_width=32,scout_range=22,channel_num=3),ScoutStaticsticVec())
        self._init_obs_space()

    def _reset(self):
        obs = self.env._reset()
        self._obs[0].reset(self.env)
        self._obs[1].reset(self.env)
        self._walkaroundIndicator = False
        self._backIndicator = False
        obs = self.observation(obs,action=0)
        # print('statistic vec', obs[1])
        return obs

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        self._walkaroundIndicator = info['walkaround']
        self._backIndicator = info['back']
        obs = self.observation(obs,action)
        # print("current img_obs",(10*obs[0][:,:,0]).astype(np.int32))
        # print("current img_obs",np.shape(obs[0][:,:,1]))#.shape)
        # print('statistic vec', obs[1])
        return obs, rwd, done, info

    def _init_obs_space(self):
        vec_dim = self._obs[0].obs_space().shape[0]*self._obs[0].obs_space().shape[1]\
                  *self._obs[0].obs_space().shape[2] + self._obs[1].obs_space().shape[0]
        low = -np.ones(vec_dim)
        high = np.ones(vec_dim)
        self.observation_space = Box(low,high)
        print("obs space",self.observation_space)

        # self.observation_space = (self._obs[0].obs_space(),self._obs[1].obs_space())
        # print('img obs space=', self._obs[0].obs_space())
        # print('statistic obs space',self._obs[1].obs_space())


        # self.observation_space = self._obs[0].obs_space()
        # print('img obs space=', self._obs[0].obs_space())

    def observation(self, obs,action):
        img_features = self._obs[0].extract(self.env, obs, self._walkaroundIndicator,self._backIndicator)
        statistic_features = self._obs[1].extract(self.env,obs,action)
        img_features_vec_like = img_features.flatten()
        return np.hstack([img_features_vec_like,statistic_features])
        # return (img_features,statistic_features)
