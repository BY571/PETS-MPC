import gym
import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import MBReplayBuffer
import glob
from utils import save, collect_random
import random
from MPC import MPC, CEM, PDDM
from model import MBEnsemble
from utils import evaluate
from tqdm import tqdm

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="PETS-MPC", help="Run name, default: PETS-MPC")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--episode_length", type=int, default=500, help="Length of one episode, default: 1000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    
    ## MB params
    parser.add_argument("--mb_buffer_size", type=int, default=100_000, help="")
    parser.add_argument("--ensembles", type=int, default=7, help="")
    parser.add_argument("--probabilistic", type=int, default=1, help="")
    parser.add_argument("--elite_size", type=int, default=5, help="")
    parser.add_argument("--hidden_size", type=int, default=200, help="")
    parser.add_argument("--hidden_layer", type=int, default=2, help="")
    parser.add_argument("--mb_lr", type=float, default=1e-3, help="")
    parser.add_argument("--loss_type", type=str, choices=["mse", "maximum_likelihood"], default="maximum_likelihood", help="")
    parser.add_argument("--update_frequency", type=int, default=250, help="")
    parser.add_argument("--rollout_select", type=str, default="random", choices=["random", "mean"], help="Define how the rollouts are composed, randomly from a random selected member of the ensemble or as the mean over all ensembles, default: random")

    #MPC params
    parser.add_argument("--mpc_type", type=str, default="cem", choices=["random", "cem", "pddm"], help="")
    parser.add_argument("--n_planner", type=int, default=500, help="") # 1000
    parser.add_argument("--depth", type=int, default=30, help="") # 30
    parser.add_argument("--action_noise", type=int, default=0, help="")
    # cem specific
    parser.add_argument("--iter_update_steps", type=int, default=3, help="")
    parser.add_argument("--k_best", type=int, default=5, help="")
    # pddm specific
    parser.add_argument("--pddm_gamma", type=float, default=1.0, help="")
    parser.add_argument("--pddm_beta", type=float, default=0.5, help="")

    

    
    args = parser.parse_args()
    return args 


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make(config.env)
    evaluation_env = gym.make(config.env)
    env.seed(config.seed)
    evaluation_env.seed(config.seed)
    
    state_size = evaluation_env.observation_space.shape[0]
    action_size = evaluation_env.action_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    
    with wandb.init(project="PETS", name=config.run_name, config=config):
        
        if config.mpc_type == "random":
            mpc = MPC(evaluation_env.action_space, n_planner=config.n_planner, depth=config.depth, device=device)
        elif config.mpc_type == "cem":
            mpc = CEM(action_space=evaluation_env.action_space,
                      n_planner=config.n_planner,
                      horizon=config.depth,
                      iter_update_steps=config.iter_update_steps,
                      k_best=config.k_best,
                      device=device)
        elif config.mpc_type == "pddm":
            mpc = PDDM(action_space=evaluation_env.action_space,
                       n_planner=config.n_planner,
                       horizon=config.depth,
                       gamma=config.pddm_gamma,
                       beta=config.pddm_beta,
                       device=device)
        else:
            raise NotImplementedError
        ensemble = MBEnsemble(state_size=state_size,
                              action_size=action_size,
                              config=config,
                              device=device)    
        print(ensemble.dynamics_model)
        mb_buffer = MBReplayBuffer(buffer_size=config.mb_buffer_size,
                                   device=device)

        collect_random(env=evaluation_env, dataset=mb_buffer, num_samples=5000)
        if config.log_video:
            evaluation_env = gym.wrappers.Monitor(evaluation_env, './video', video_callable=lambda x: x%10==0, force=True)

        # do training
        for i in tqdm(range(1, config.episodes+1)):
            state = env.reset()
            episode_steps = 0
            while episode_steps < config.episode_length:

                if steps % config.update_frequency == 0:
                    train_inputs, train_labels = mb_buffer.get_dataloader()
                    losses, trained_epochs = ensemble.train(train_inputs, train_labels)           
                    wandb.log({"Episode": i, "MB mean loss": np.mean(losses), "MB mean trained epochs": trained_epochs}, step=steps)
                    tqdm.write("\rEpisode: {} | Ensemble losses: {}".format(i, losses))

                action = mpc.get_next_action(state, ensemble, noise=config.action_noise, probabilistic=config.probabilistic)
                
                next_state, reward, done, _ = env.step(action)

                mb_buffer.add(state, action, reward, next_state, done)

                state = next_state
                episode_steps += 1
                steps += 1
                if done:
                    state = env.reset()

            # do evaluation runs 
            rewards = evaluate(evaluation_env, mpc, ensemble)
            average10.append(rewards)
            tqdm.write("\rEpisode: {} | Reward: {} | Steps: {}".format(i, rewards, steps,))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": steps,
                       "Episode": i,
                       "Env Buffer size": mb_buffer.__len__()})

            # log evaluation runs to wandb
            if config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})


if __name__ == "__main__":
    config = get_config()
    train(config)
