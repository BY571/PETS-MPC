import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

    
class Ensemble_FC_Layer(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size, bias=True):
        super(Ensemble_FC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        pass


    def forward(self, x) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    
class DynamicsModel(nn.Module):

    def __init__(self, state_size, action_size, ensemble_size=7, hidden_layer=3, hidden_size=200, lr=1e-2, device="cpu"):
        super(DynamicsModel, self).__init__()
        self.ensemble_size = ensemble_size
        activation = nn.SiLU() # nn.RELU()
        self.input_layer = Ensemble_FC_Layer(state_size + action_size, hidden_size, ensemble_size)
        hidden_layers = []
        hidden_layers.append(activation)
        for _ in range(hidden_layer):
            hidden_layers.append(Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size))
            hidden_layers.append(activation)
        self.hidden_layers = nn.Sequential(*hidden_layers)
 
        self.mu = Ensemble_FC_Layer(hidden_size, state_size + 1, ensemble_size)
        self.log_std = Ensemble_FC_Layer(hidden_size, state_size + 1, ensemble_size)

        self.min_logstd = (- torch.ones((1, state_size + 1)).float() * 10).to(device)
        self.max_logstd = (torch.ones((1, state_size + 1)).float() * 2).to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_f = nn.MSELoss(reduction='none')
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
    
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.min_logstd, self.max_logstd)

        return mu, log_std
    
    def calc_loss(self, inputs, targets, validate=False):
        mu, log_std = self(inputs)
        assert mu.shape[1:] == targets.shape[1:]
        prediction = torch.normal(mu, log_std.exp())
        assert prediction.shape == targets.shape
        if not validate:
            loss = self.loss_f(prediction, targets).mean() #mean(-1).mean(-1).sum()
            return loss
        else:
            loss = self.loss_f(prediction, targets).mean(-1).mean(-1)
            return loss

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
       

class MBEnsemble():
    def __init__(self, state_size, action_size, config, device):
                
        self.device = device
        self.ensemble = []
        self.probabilistic = True
        self.batch_size = config.batch_size
        self.validation_percentage = 0.2
        self.n_ensembles = config.ensembles
        self.dynamics_model = DynamicsModel(state_size=state_size,
                                      action_size=action_size,
                                      ensemble_size=config.ensembles,
                                      hidden_layer=config.hidden_layer,
                                      hidden_size=config.hidden_size,
                                      lr=config.mb_lr,
                                      device=device).to(device)

        self.elite_size = config.elite_size
        self.elite_idxs = []
        
        self.max_not_improvements = 5
        self._current_best = [1e10 for i in range(self.n_ensembles)]
        self.improvement_threshold = 0.01
        self.break_counter = 0
        self.env_name = config.env
        
    def train(self, inputs, labels):
        losses = 0
        epochs_trained = 0
        self.break_counter = 0
        break_training = False
        
        num_validation = int(inputs.shape[0] * self.validation_percentage)
        train_inputs, train_labels = inputs[num_validation:], labels[num_validation:]
        holdout_inputs, holdout_labels = inputs[:num_validation], labels[:num_validation]
                
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(self.device)
        holdout_inputs = holdout_inputs[None, :, :].repeat(self.n_ensembles, 1, 1)
        holdout_labels = holdout_labels[None, :, :].repeat(self.n_ensembles, 1, 1)
        
        num_training_samples = train_inputs.shape[0]
        while True:
            train_idx = np.vstack([np.random.permutation(num_training_samples) for _ in range(self.n_ensembles)])
            
            self.dynamics_model.train()
            for start_pos in range(0, num_training_samples, self.batch_size):
                idx = train_idx[:, start_pos: start_pos + self.batch_size]
                train_input = train_inputs[idx]
                train_label = train_labels[idx]
                train_input = torch.from_numpy(train_input).float().to(self.device)
                train_label = torch.from_numpy(train_label).float().to(self.device)
                loss = self.dynamics_model.calc_loss(train_input, train_label)
                self.dynamics_model.optimize(loss)
                epochs_trained += 1
                
            # evaluation
            self.dynamics_model.eval()
            with torch.no_grad():
                val_losses = self.dynamics_model.calc_loss(holdout_inputs, holdout_labels, validate=True)
                val_losses = val_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(losses)
                self.elite_idxs = sorted_loss_idx[:self.elite_size].tolist()
                break_training = self.test_break_condition(val_losses)
                if break_training:
                    break
            
        assert len(val_losses) == self.n_ensembles, f"epoch_losses: {len(val_losses)} =/= {self.n_ensembles}"
        
        return val_losses, np.mean(epochs_trained)
    
    def test_break_condition(self, current_losses):
        keep_train = False
        for i in range(len(current_losses)):
            current_loss = current_losses[i]
            best_loss = self._current_best[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > self.improvement_threshold:
                self._current_best[i] = current_loss
                keep_train = True
    
        if keep_train:
            self.break_counter = 0
        else:
            self.break_counter += 1
        if self.break_counter >= self.max_not_improvements:
            return True
        else:
            return False


    def run_ensemble_prediction(self, states, actions):
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        inputs = torch.cat((states, actions), axis=-1).to(self.device)
        inputs = inputs[None, :, :].repeat(self.n_ensembles, 1, 1)
        with torch.no_grad():
            mus, var = self.dynamics_model(inputs, return_log_var=False)

        # [ensembles, batch, prediction_shape]
        assert mus.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        assert var.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        return mus.cpu().numpy(), var.cpu().numpy()
            
    