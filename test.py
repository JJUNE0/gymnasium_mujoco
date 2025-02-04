import os
import glob

import gymnasium as gym
import torch
import argparse

from algorithm.PPO.trainer import  Trainer
from algorithm.PPO.PPO import PPO

def get_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, default='td3',
                        help='select an algorithm among td3, ppo, sac')

    parser.add_argument('--checkpoint_freq', default=20, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoint/')

    # --는 필수 인자가 아닌 선택적 인자.
    # InvertedPendulum-v5
    # HalfCheetah-v5
    parser.add_argument('--env-name', default='HalfCheetah-v5')
    parser.add_argument('--random-seed', default=336699, type=int)
    #parser.add_argument('--n-history', default=3, type=int)
    parser.add_argument('--max-episode', default=5000, type=int)

    #parser.add_argument('--eval_flag', default=True, type=bool)
    #parser.add_argument('--eval-freq', default=5000, type=int)
    #parser.add_argument('--eval-episode', default=5, type=int)
    #parser.add_argument('--render', default=True, type=bool)

    parser.add_argument('--start-step', default=200, type=int)
    parser.add_argument('--max-step', default=5000000, type=int)
    parser.add_argument('--update_after', default=2000, type=int)

    parser.add_argument('--hidden-dims', default=(400, 300))
    parser.add_argument('--hidden-size', default=64)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int)
    parser.add_argument('--update-every', default=50, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=0.0001, type=float)  # 3e-4
    parser.add_argument('--critic-lr', default=0.0001, type=float)  # 3e-4
    parser.add_argument('--tau', default=0.001, type=float)

    param = parser.parse_args()

    return param

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.algo == 'td3':
        from algorithm.TD3.TD3 import TD3 as Agent
    elif args.algo == 'ppo':
        from algorithm.PPO.PPO import PPO as Agent
    elif args.algo == 'sac':
        from algorithm.SAC.SAC import SAC as Agent

    env = gym.make(args.env_name, render_mode = 'human')
    agent = Agent(env, device, args)
    agent.load_weights(args.checkpoint_dir+args.algo+'/')


    max_episodes = 1000
    episdoe = 0
    episode_reward = 0
    while episdoe < max_episodes:

        state, _ = env.reset()
        done = False
        while not done:

            action = agent.get_action(state, evaluation=True,test=True)

            state,reward,done,_,_ = env.step(action)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

            if done:
                episdoe+=1
                print('episdoe : ', episdoe, 'episode_reward:', episode_reward)
                episode_reward = 0

if __name__ == "__main__":
    args = get_parameters()
    main(args)