from sc2scout.wrapper.reward import evade_img_reward as er
from sc2scout.wrapper.reward import scout_reward as sr
from sc2scout.wrapper.explore_enemy.reward_wrapper import ScoutExploreEvadeRwd

class ExploreWithEvadeRwdWrapper(ScoutExploreEvadeRwd):
    def __init__(self, env):
        super(ExploreWithEvadeRwdWrapper, self).__init__(env)

    def _assemble_reward(self):
        self._forward_rewards = [sr.HomeReward(negative=False,weight=1),
                                 sr.EnemyBaseReward(negative=False,weight=1),
                                 sr.ViewEnemyReward(weight=10),
                                 # sr.EnemyBaseArrivedReward(weight=50),
                                 sr.MinDistReward(negative=True),
                                 # er.EvadeDistanceReward(weight=1),
                                 er.EvadeUnderAttackRwd(weight=1),
                                 # er.EnemyInRangeRwd(weight=1),
                                 # sr.AreaOfOverlapReward(weight=2)
                                 ]

        self._explore_rewards = [
                                 # sr.HomeReward(negative=True,weight=1),
                                 # sr.ViewEnemyResourcesAndBase(weight=10),
                                 sr.ExploreStateRwd(weight=20),
                                 # er.EvadeDistanceReward(weight=1),
                                 er.EvadeUnderAttackRwd(weight=1),
                                 sr.ViewEnemyResourcesAndBase(weight=5),
                                 # er.EnemyInRangeRwd(weight=1),
                                 sr.ExploreAcclerateRwd(weight=0.5)
                                ]

        self._backward_rewards = [sr.HomeReward(back=True, negative=False,weight=1),
                                  #sr.EnemyBaseReward(back=True, negative=False,weight=1),
                                  sr.HomeArrivedReward(weight=50),
                                  #sr.MinDistReward(negative=True),
                                  sr.BackwardStateRwd(weight=20),
                                  # er.EvadeDistanceReward(weight=1),
                                  er.EvadeUnderAttackRwd(weight=1),
                                  # er.EnemyInRangeRwd(weight=1),
                                  # sr.AreaOfOverlapReward(weight=2),
                                  # sr.HitEnemyBaseReward(weight=50),
                                  ]

        self._final_rewards = [sr.RoundTripFinalReward(weight=100),
                               # er.EvadeFinalRwd(weight=50)
                               ]




