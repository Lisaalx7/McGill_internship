""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0
import mujoco as mj


class BackEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'pose_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 5.0,
        "act_reg": .00,
        "penalty": 0,
        "done": 0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)

    def _setup(self,
            viz_site_targets:tuple = None,  # site to use for targets visualization []
            target_jnt_range:dict = None,   # joint ranges as tuples {name:(min, max)}_nq
            target_jnt_value:list = None,   # desired joint vector [des_qpos]_nq
            reset_type = "init",            # none; init; random
            target_type = "fixed",       # generate; switch; fixed
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            pose_thd = 0.15,
            weight_bodyname = None,
            weight_range = None,
            **kwargs,
        ):
        self.reset_type = reset_type
        self.target_type = target_type
        self.pose_thd = pose_thd
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range

        # resolve joint demands
        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value

        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=viz_site_targets,
                **kwargs,
                )
        #self.init_qpos = self.sim.model.key_qpos[0]

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        self.obs_dict['pose_err'] = np.array([
            self.sim.data.joint('flex_extension').qpos.copy(),
            self.sim.data.joint('lat_bending').qpos.copy(),
            self.sim.data.joint('axial_rotation').qpos.copy()]) - np.array([0, 0, 0])
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        """
        for i in range(sim.model.njnt):
            # Get the starting index for the joint name in the `names` array
            start_idx = sim.model.name_jntadr[i]
            # Extract the name
            name = ''
            while sim.model.names[start_idx] != 0:  # 0 is the null terminator in the names array
                name += chr(sim.model.names[start_idx])
                start_idx += 1
            print(f"Joint {i}: {name}")
        """
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        obs_dict['act'] = sim.data.act[:].copy() if sim.model.na>0 else np.zeros_like(obs_dict['qpos'])
        #obs_dict['pose_err'] = self.target_jnt_value[:18] - obs_dict['qpos'][:18]
        obs_dict['pose_err'] = np.array([
            sim.data.joint('flex_extension').qpos.copy(),
            sim.data.joint('lat_bending').qpos.copy(),
            sim.data.joint('axial_rotation').qpos.copy()]) - np.array([0, 0, 0])
                #print('compare', self.target_jnt_value, obs_dict['qpos'])
        #print([self.sim.data.joint('flex_extension').qpos.copy(),self.sim.data.joint('lat_bending').qpos.copy(), self.sim.data.joint('axial_rotation').qpos.copy()])
        
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'][0][0][0], axis=-1) #np.linalg.norm(obs_dict['pose_err'], axis=-1)
        act_mag = np.linalg.norm(obs_dict['act'], axis=-1)
        if self.sim.model.na !=0: act_mag= act_mag/self.sim.model.na
        far_th = np.pi/2

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',     -pose_dist),
            ('bonus',    1 * (pose_dist<self.pose_thd) + 1.*(pose_dist<1.5*self.pose_thd)),
            ('penalty', -1 * (pose_dist>far_th)),
            ('act_reg', -1 * act_mag[0][0]),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<self.pose_thd),
            ('done',    pose_dist>far_th),
        ))

        #print(self.dt)
        #print([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, **kwargs):
        # update init state
        if self.reset_type is None or self.reset_type == "none":
            # no reset; use last state
            ## NOTE: fatigue is also not reset in this case!
            obs = self.get_obs()
        elif self.reset_type == "init":
            # reset to init state
            obs = super().reset(**kwargs)
        elif self.reset_type == "random":
            # reset to random state
            jnt_init = self.np_random.uniform(high=self.sim.model.jnt_range[:,1], low=self.sim.model.jnt_range[:,0])
            obs = super().reset(reset_qpos=jnt_init, **kwargs)
        else:
            print("Reset Type not found")

        return obs