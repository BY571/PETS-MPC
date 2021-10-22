import gym
import numpy as np
import torch
from copy import deepcopy


class RandomPolicy():
    def __init__(self, action_space, n_planner, device=None) -> None:
        self.n_planner = n_planner
        self.device = device
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_space = action_space.n
            self.action_type = "discrete"
        elif type(action_space) == gym.spaces.box.Box:
            self.action_space = action_space.shape[0]
            self.action_type = "continuous"
            self.action_low = action_space.low
            self.action_high = action_space.high
        else:
            raise ValueError ("Given action space does not exist!")

    def get_actions(self, states):
        if self.action_type == "discrete":
            actions = torch.randint(self.action_space, size=(self.n_planner, 1)).to(self.device)
            return actions
        else:
            actions = np.random.uniform(low=self.action_low,
                                        high=self.action_high,
                                        size=(self.n_planner, self.action_space))
            return actions


class MPC():
    def __init__(self, action_space, n_planner=1, depth=10, device="cpu") -> None:
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_type = "discrete"
        elif type(action_space) == gym.spaces.box.Box:
            self.action_type = "continuous"
        else:
            raise ValueError ("Unknonw action space")
        self.n_planner = n_planner
        self.depth = depth
        self.device = device
        self.policy = RandomPolicy(action_space=action_space,
                                   n_planner=n_planner,
                                   device=device)

    def get_next_action(self, state, model, noise=False, probabilistic=True):

        states = np.repeat(state.reshape(1, state.shape[0]), self.n_planner, axis=0)

        actions, returns = self.run_mpc(states, model, probabilistic)
        optimal_action = actions[returns.argmax()]

        if noise and self.action_type=="continuous":
            optimal_action += np.random.normal(0, 0.005, size=optimal_action.shape)
        if self.action_type == "discrete":
            optimal_action = optimal_action[0]
        return optimal_action

    def run_mpc(self, states, model, probabilistic=True):
        
        returns = torch.zeros((self.n_planner,1))
        for i in range(self.depth):
            actions = self.policy.get_actions(states)

            with torch.no_grad():
                ensemble_means, ensemble_stds = model.run_ensemble_prediction(states, actions)
                ensemble_means[:, :, :-1] += states
                ensemble_means = ensemble_means.mean(0)
                ensemble_stds = np.sqrt(ensemble_stds).mean(0)
                
                if probabilistic:
                    predictions = ensemble_means + np.random.normal(size=ensemble_means.shape) * ensemble_stds
                else:
                    predictions = ensemble_means

            states = predictions[:, :-1]
            
            returns += predictions[:, -1]
            if i == 0:
                first_actions = deepcopy(actions)

        return first_actions, returns
