import gym
import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer, MBReplayBuffer
import glob
from utils import save, collect_random
import random
from MPC import MPC
from model import MBEnsemble
import multipro
from utils import evaluate

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="PETS-MPC", help="Run name, default: PETS-MPC")
    parser.add_argument("--env", type=str, default="Pendulum-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--episode_length", type=int, default=1000, help="Length of one episode, default: 1000")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Maximal training dataset size, default: 1_000_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--npolicy_updates", type=int, default=20, help="")
    parser.add_argument("--parallel_envs", type=int, default=1, help="Number of parallel environments, default: 1")
    
    ## MB params
    parser.add_argument("--mb_buffer_size", type=int, default=100_000, help="")
    parser.add_argument("--n_rollouts", type=int, default=400, help="")
    parser.add_argument("--ensembles", type=int, default=4, help="")
    parser.add_argument("--hidden_size", type=int, default=200, help="")
    parser.add_argument("--mb_lr", type=float, default=1e-2, help="")
    parser.add_argument("--update_frequency", type=int, default=250, help="")
    parser.add_argument("--rollout_select", type=str, default="random", choices=["random", "mean"], help="Define how the rollouts are composed, randomly from a random selected member of the ensemble or as the mean over all ensembles, default: random")

    #MPC params
    parser.add_argument("--n_planner", type=int, default=5000, help="")
    parser.add_argument("--depth", type=int, default=32, help="")
    
    args = parser.parse_args()
    return args 


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    #envs = multipro.SubprocVecEnv([lambda: gym.make(config.env) for i in range(config.parallel_envs)])
    env = gym.make(config.env)
    evaluation_env = gym.make(config.env)
    env.seed(config.seed)
    evaluation_env.seed(config.seed)
    
    state_size = evaluation_env.observation_space.shape[0]
    action_size = evaluation_env.action_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    with wandb.init(project="PETS", name=config.run_name, config=config):
        
        mpc = MPC(evaluation_env.action_space, n_planner=config.n_planner, depth=config.depth, device=device)
        
        ensemble = MBEnsemble(state_size=state_size,
                              action_size=action_size,
                              config=config,
                              device=device)    
                
        mb_buffer = MBReplayBuffer(buffer_size=config.mb_buffer_size,
                                   batch_size=config.n_rollouts,
                                   device=device)

        collect_random(env=evaluation_env, dataset=mb_buffer, num_samples=2500)
        if config.log_video:
            evaluation_env = gym.wrappers.Monitor(evaluation_env, './video', video_callable=lambda x: x%10==0, force=True)

        # do training
        for i in range(1, config.episodes+1):
            state = env.reset()
            episode_steps = 0
            while episode_steps < config.episode_length:

                if total_steps % config.update_frequency == 0:
                    train_dataloader = mb_buffer.get_dataloader(batch_size=256)
                    loss = ensemble.train(train_dataloader)
                    wandb.log({"Episode": i, "MB Loss": loss}, step=steps)                

                action = mpc.get_next_action(state, ensemble)
                
                next_state, reward, done, _ = env.step(action)

                mb_buffer.add(state, action, reward, next_state, done)

                state = next_state
                episode_steps += config.parallel_envs
                steps += 1
                total_steps += 1
                if done:
                    state = env.reset()

            # do evaluation runs 
            rewards = evaluate(evaluation_env, mpc, ensemble)
            average10.append(rewards)
            print("Episode: {} | Reward: {} | Steps: {}".format(i, rewards, steps,))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Steps": steps,
                       "Episode": i,
                       "Env Buffer size": mb_buffer.__len__()})

            # log evaluation runs to wandb
            if config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            # if i % config.save_every == 0:
            #     save(config, save_name="MBPO-SAC", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
