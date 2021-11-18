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

class CEM():
    def __init__(self, action_space, n_planner=500, horizon=16, iter_update_steps=5, k_best=10, device=None)-> None:
        self.n_planner = n_planner
        self.k_best = k_best
        self.device = device
        self.update_alpha = 0.01
        self.action_space = action_space.shape[0]
        self.horizon = horizon
        self.iter_update_steps = iter_update_steps
        self.mu = np.zeros(self.action_space)
        self.var = np.ones(self.action_space) # var = sigma² , sigma = std
        
    def get_next_action(self, initial_state, model, noise=False, probabilistic=True):
        initial_state = np.repeat(initial_state[None, :], self.n_planner, 0)
        for i in range(self.iter_update_steps):

            states = initial_state
            reward_summed = np.zeros((self.n_planner, 1))
            action_history = []
            for i in range(self.horizon):
                actions = np.random.normal(self.mu, np.sqrt(self.var), size=(states.shape[0], self.action_space))
                actions += np.random.normal(0, 0.01, size=actions.shape)

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

                reward_summed += predictions[:, -1][:, None]
                action_history.append(actions[None, :]) # shape (horizon, n_planner, action_space)
            
            k_best_rewards, k_best_actions = self.select_k_best(reward_summed, action_history)
            self.update_gaussians(k_best_actions)
        
        best_action = k_best_actions[0, -1, :]#[None, :] # 0 element in the horizon, -1 best planner
        
        assert best_action.shape == (self.action_space,)
        return best_action
            
    
    def select_k_best(self, rewards, action_hist):
        assert rewards.shape == (self.n_planner, 1)
        idxs = np.argsort(rewards, axis=0)

        action_hist = np.concatenate(action_hist) # shape (horizon, n_planner, action_space)
        sorted_actions_hist = action_hist[:, idxs, :].squeeze(2) # sorted (horizon, n_planner, action_space)

        k_best_actions_hist = sorted_actions_hist[:, -self.k_best:, :]
        k_best_rewards = rewards[idxs].squeeze(1)[-self.k_best:]
        assert k_best_rewards.shape == (self.k_best, 1)
        assert k_best_actions_hist.shape == (self.horizon, self.k_best, self.action_space)
        return k_best_rewards, k_best_actions_hist


    def update_gaussians(self, best_actions):
        assert best_actions.shape == (self.horizon, self.k_best, self.action_space)
        new_mu = best_actions.mean(0).mean(0)
        new_var = best_actions.var(0).var(0)
        old_mu = deepcopy(self.mu)
        old_var = deepcopy(self.var)
        self.mu = (self.update_alpha * old_mu + (1.0-self.update_alpha) * new_mu)
        #print("old mu: {} new_mu: {} updated mu: {}".format(old_mu, new_mu, self.mu))
        self.var = (self.update_alpha * old_var + (1.0 - self.update_alpha)*new_var)


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
