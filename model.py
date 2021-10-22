import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

    
class DynamicsModel(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=200, seed=1, lr=1e-3, device="cpu"):
        super(DynamicsModel, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, state_size + 1)
        self.log_var = nn.Linear(hidden_size, state_size + 1)
        
        self.activation = nn.SiLU()

        self.min_logvar = nn.Parameter((-torch.ones((1, state_size + 1)).float() * 10).to(device), requires_grad=False)
        self.max_logvar = nn.Parameter((torch.ones((1, state_size + 1)).float() / 2).to(device), requires_grad=False)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state, action, return_log_var=False):
        x = torch.cat((state, action), dim=-1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        
        mu = self.mu(x)

        log_var = self.max_logvar - F.softplus(self.max_logvar - self.log_var(x))
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        if return_log_var:
            return mu, log_var
        else:
            return mu, torch.exp(log_var)
    
    def calc_loss(self, state, action, targets, include_var=True):
        mu, log_var = self(state, action, return_log_var=True)
        assert mu.shape == targets.shape
        if include_var:
            inv_var = (-log_var).exp()
            loss = ((mu - targets)**2 * inv_var).mean(-1).mean(-1) + log_var.mean(-1).mean(-1)
            return loss
        else:
            return ((mu - targets)**2).mean(-1).mean(-1)
            

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss.backward()
        self.optimizer.step()

       

class MBEnsemble():
    def __init__(self, state_size, action_size, config, device):
                
        self.device = device
        self.ensemble = []
        self.probabilistic = True
        self.n_ensembles = config.ensembles
        for i in range(self.n_ensembles):
            dynamics = DynamicsModel(state_size=state_size,
                                               action_size=action_size,
                                               hidden_size=config.hidden_size,
                                               lr=config.mb_lr,
                                               seed=i,
                                               device=device).to(device)
            self.ensemble.append(dynamics)          

        self.rollout_select = config.rollout_select
        self.elite_size = config.elite_size
        self.elite_idxs = []
        
        self.max_not_improvements = 5
        self.improvement_threshold = 0.01
        self.break_counter = 0
        self.env_name = config.env
        
    def train(self, train_loader, test_loader):
        losses = np.zeros(len(self.ensemble))
        epochs_trained = np.zeros(len(self.ensemble))
        for idx, model in enumerate(self.ensemble):
            best_val_loss = 10_000 # not elegant 
            self.break_counter = 0
            break_training = False
            while True:
                model.train()
                for (s, a, r, ns, d) in train_loader:
                    delta_state = ns - s
                    targets = torch.cat((delta_state, r), dim=-1).to(self.device)
                    loss = model.calc_loss(s, a, targets)
                    model.optimize(loss)
                    epochs_trained[idx] += 1
                    
                # evaluation
                model.eval()
                with torch.no_grad():
                    for (s, a, r, ns, d) in test_loader:
                        delta_state = ns - s
                        targets = torch.cat((delta_state, r), dim=-1).to(self.device)
                        loss = model.calc_loss(s, a, targets, include_var=False)
                        losses[idx] = loss.item()
                    break_training, best_val_loss = self.test_break_condition(loss.item(), best_val_loss)
                if break_training:
                    break

            
        assert len(losses) == self.n_ensembles, f"epoch_losses: {len(losses)} =/= {self.n_ensembles}"
        sorted_loss_idx = np.argsort(losses)
        self.elite_idxs = sorted_loss_idx[:self.elite_size].tolist()
        
        return losses, np.mean(epochs_trained)
    
    def test_break_condition(self, current_loss, best_loss):
        keep_train = False
        improvement = (best_loss - current_loss) / best_loss
        if improvement > self.improvement_threshold:
            best_loss = current_loss
            keep_train = True
        if keep_train:
            self.break_counter = 0
        else:
            self.break_counter += 1
        if self.break_counter >= self.max_not_improvements:
            return True, best_loss
        else:
            return False, best_loss
            
    
    
    def run_ensemble_prediction(self, states, actions):
        mus_list = []
        stds_list = []
        with torch.no_grad():
            for model in self.ensemble:
                mus, stds = model(torch.from_numpy(states).float().to(self.device),
                                        torch.from_numpy(actions).float().to(self.device), return_log_var=False)
                mus_list.append(mus.unsqueeze(0))
                stds_list.append(stds.unsqueeze(0))
        all_mus = torch.cat(mus_list, axis=0)
        all_stds = torch.cat(stds_list, axis=0)
        # [ensembles, batch, prediction_shape]
        assert all_mus.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        assert all_stds.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        return all_mus.cpu().numpy(), all_stds.cpu().numpy()
            
    