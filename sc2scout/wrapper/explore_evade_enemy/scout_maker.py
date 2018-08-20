from sc2scout.wrapper.wrapper_factory import WrapperMaker
from sc2scout.wrapper.explore_evade_enemy.explore_with_evade_obs_wrapper import ExploreWithEvadeObsWrapper
from sc2scout.wrapper.explore_evade_enemy.explore_with_evade_rew_wrapper import ExploreWithEvadeRwdWrapper
from sc2scout.wrapper.explore_evade_enemy.explore_with_evade_terminal_wrapper import ExploreWithEvadeTerminalWrapper
from sc2scout.wrapper.explore_enemy.action_wrapper import ZergScoutActWrapper
from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper2,ZergScoutWrapper

from baselines import deepq

class ScoutMakerV0(WrapperMaker):
    def __init__(self):
        super(ScoutMakerV0, self).__init__('scout_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = ExploreWithEvadeTerminalWrapper(env)
        env = ExploreWithEvadeRwdWrapper(env)
        env = ExploreWithEvadeObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.cnn_to_mlp(
            convs=[(6, 5, 1), (16, 5, 1), (120, 5, 1)],
            hiddens=[128])