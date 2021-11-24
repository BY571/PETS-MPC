import gym
import numpy as np
import torch
from copy import copy, deepcopy
import scipy.stats as stats


class RandomPolicy():
    def __init__(self, action_space, n_planner, device=None) -> None:
        self.n_planner = n_planner
        self.device = device
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_space = action_space.n
            self.action_type = "discrete"
            self.action_low = None
            self.action_high = None
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
        self.action_space = action_space.shape[0]
        self.horizon = horizon
        self.iter_update_steps = iter_update_steps
        self.update_alpha = 0.0
        self.epsilon = 0.001
        self.ub = 1
        self.lb = -1
        
    def get_next_action(self, initial_state, model, noise=False, probabilistic=True):
        initial_state = np.repeat(initial_state[None, :], self.n_planner, 0)
        mu = np.zeros(self.horizon*self.action_space)
        var = 5 * np.ones(self.horizon*self.action_space)
        X = stats.truncnorm(self.lb, self.ub, loc=np.zeros_like(mu), scale=np.ones_like(mu))
        i = 0
        while ((i < self.iter_update_steps) and (np.max(var) > self.epsilon)):
            states = initial_state
            reward_summed = np.zeros((self.n_planner, 1))
            #variables
            lb_dist = mu - self.lb
            ub_dist = self.ub - mu
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            
            actions = X.rvs(size=[self.n_planner, self.horizon*self.action_space]) * np.sqrt(constrained_var) + mu
            actions_t = np.clip(actions, -1, 1).reshape(self.n_planner, self.horizon, self.action_space)
            for t in range(self.horizon):
                with torch.no_grad():
                    ensemble_means, ensemble_stds = model.run_ensemble_prediction(states, actions_t[:, t, :])
                    ensemble_means[:, :, :-1] += states
                    ensemble_means = ensemble_means.mean(0)
                    ensemble_stds = np.sqrt(ensemble_stds).mean(0)
                    
                    if probabilistic:
                        predictions = ensemble_means + np.random.normal(size=ensemble_means.shape) * ensemble_stds
                    else:
                        predictions = ensemble_means

                states = predictions[:, :-1]
                reward_summed += predictions[:, -1][:, None]
            
            k_best_rewards, k_best_actions = self.select_k_best(reward_summed, actions)
            mu, var = self.update_gaussians(mu, var, k_best_actions)
            i += 1
        
        best_action_sequence = mu.reshape(self.horizon, -1)
        best_action = np.copy(best_action_sequence[-1])
        assert best_action.shape == (self.action_space,)
        return best_action
            
    
    def select_k_best(self, rewards, action_hist):
        assert rewards.shape == (self.n_planner, 1)
        idxs = np.argsort(rewards, axis=0)

        elite_actions = action_hist[idxs][-self.k_best:, :].squeeze(1) # sorted (elite, horizon x action_space)
        k_best_rewards = rewards[idxs][-self.k_best:, :].squeeze(-1)

        assert k_best_rewards.shape == (self.k_best, 1)
        assert elite_actions.shape == (self.k_best, self.horizon*self.action_space)
        return k_best_rewards, elite_actions


    def update_gaussians(self, old_mu, old_var, best_actions):
        assert best_actions.shape == (self.k_best, self.horizon*self.action_space)

        new_mu = best_actions.mean(0)
        new_var = best_actions.var(0)

        mu = (self.update_alpha * old_mu + (1.0 - self.update_alpha) * new_mu)
        var = (self.update_alpha * old_var + (1.0 - self.update_alpha) * new_var)
        assert mu.shape == (self.horizon*self.action_space, )
        assert var.shape == (self.horizon*self.action_space, )
        return mu, var


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


class PDDM():
    def __init__(self, action_space, n_planner=200, horizon=7, gamma=10, beta=0.5, device=None)-> None:
        self.n_planner = n_planner
        self.device = device
        self.action_space = action_space.shape[0]
        self.horizon = horizon
        self.gamma = gamma
        self.beta = beta
        self.mu = np.zeros((self.horizon, self.action_space))
        
    def get_next_action(self, initial_state, model, noise=False, probabilistic=True):
        initial_state = np.repeat(initial_state[None, :], self.n_planner, 0)
        actions, returns = self.get_pred_trajectories(initial_state, model, probabilistic)
        optimal_action = self.update_mu(actions, returns)
       
        if noise:
            optimal_action += np.random.normal(0, 0.005, size=optimal_action.shape)
        return optimal_action
        
    def update_mu(self, action_hist, returns):
        assert action_hist.shape == (self.n_planner, self.horizon, self.action_space)
        assert returns.shape == (self.n_planner, 1)

        c = np.exp(self.gamma * (returns) -np.max(returns))
        d = np.sum(c) + 1e-10
        assert c.shape == (self.n_planner, 1)
        assert d.shape == (), "Has shape {}".format(d.shape)
        c_expanded = c[:, :, None]
        assert c_expanded.shape == (self.n_planner, 1, 1)
        weighted_actions = c_expanded * action_hist
        self.mu = weighted_actions.sum(0) / d
        assert self.mu.shape == (self.horizon, self.action_space)       
        
        return self.mu[0]
    
    def sample_actions(self, past_action):
        u = np.random.normal(loc=0, scale=1.0, size=(self.n_planner, self.horizon, self.action_space))
        actions = u.copy()
        for t in range(self.horizon):
            if t == 0:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * past_action
            else:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * actions[:, t-1, :]
        assert actions.shape == (self.n_planner, self.horizon, self.action_space), "Has shape {} but should have shape {}".format(actions.shape, (self.n_planner, self.horizon, self.action_space))
        actions = np.clip(actions, -1, 1)
        return actions
    
    def get_pred_trajectories(self, states, model, probabilistic): 
        returns = np.zeros((self.n_planner, 1))

        np.random.seed()
        past_action = self.mu[0].copy()
        actions = self.sample_actions(past_action)
        for t in range(self.horizon):
            with torch.no_grad():
                actions_t = actions[:, t, :]
                assert actions_t.shape == (self.n_planner, self.action_space)
                ensemble_means, ensemble_stds = model.run_ensemble_prediction(states, actions_t)
                ensemble_means[:, :, :-1] += states
                ensemble_means = ensemble_means.mean(0)
                ensemble_stds = np.sqrt(ensemble_stds).mean(0)
                
                if probabilistic:
                    predictions = ensemble_means + np.random.normal(size=ensemble_means.shape) * ensemble_stds
                else:
                    predictions = ensemble_means

            states = predictions[:, :-1]
            returns += predictions[:, -1][:, None]
        return actions, returns