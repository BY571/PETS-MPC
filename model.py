from networks import DynamicsModel
import torch
import torch.optim as optim
import random
import numpy as np


class MBEnsemble():
    def __init__(self, state_size, action_size, config, device):
                
        self.device = device
        self.ensemble = []

        self.n_ensembles = config.ensembles
        for i in range(self.n_ensembles):
            dynamics = DynamicsModel(state_size=state_size,
                                               action_size=action_size,
                                               hidden_size=config.hidden_size,
                                               lr=config.mb_lr,
                                               seed=i,
                                               device=device).to(device)
            self.ensemble.append(dynamics)          

        self.n_rollouts = config.n_rollouts
        self.rollout_select = config.rollout_select

        self.elite_idxs = []
        
    def train(self, train_dataloader):
        epoch_losses = []
        for model in self.ensemble:
            for (s, a, r, ns, d) in train_dataloader:
                targets = torch.cat((ns,r), dim=-1).to(self.device)
                loss = model.calc_loss(s, a, targets)
                model.optimize(loss)
                epoch_losses.append(loss.item())
               

        return np.mean(epoch_losses)
    
    def run_ensemble_prediction(self, states, actions):
        prediction_list = []
        with torch.no_grad():
            for model in self.ensemble:
                predictions, _ = model(torch.from_numpy(states).float().to(self.device),
                                        torch.from_numpy(actions).float().to(self.device))
                prediction_list.append(predictions.unsqueeze(0))
        all_ensemble_predictions = torch.cat(prediction_list, axis=0) 
        # [ensembles, batch, prediction_shape]
        assert all_ensemble_predictions.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        return all_ensemble_predictions

            
    