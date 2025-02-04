import sys
import gymnasium as gym
import argparse
import torch
import numpy as np

def get_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, default='td3',
                        help='select an algorithm among td3, ppo, sac, dqn')

    parser.add_argument('--checkpoint_freq', default=20, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoint/')

    # --는 필수 인자가 아닌 선택적 인자.
    # InvertedPendulum-v5
    # HalfCheetah-v5
    parser.add_argument('--env-name', default='HalfCheetah-v5')
    parser.add_argument('--random-seed', default=336699, type=int)
    #parser.add_argument('--n-history', default=3, type=int)
    parser.add_argument('--max-episode', default=1000, type=int)

    #parser.add_argument('--eval_flag', default=True, type=bool)
    parser.add_argument('--eval-freq', default=5000, type=int)
    #parser.add_argument('--eval-episode', default=5, type=int)
    #parser.add_argument('--render', default=True, type=bool)

    parser.add_argument('--start-step', default=25e3, type=int)
    parser.add_argument('--max-step', default=1000000, type=int)
    parser.add_argument('--update_after', default=2000, type=int)

    parser.add_argument('--hidden-dims', default=(400, 300))
    parser.add_argument('--hidden-size', default=64)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int)
    parser.add_argument('--update-every', default=50, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=0.0003, type=float)  # 3e-4
    parser.add_argument('--critic-lr', default=0.0003, type=float)  # 3e-4
    parser.add_argument('--tau', default=0.001, type=float)

    # DQN
    parser.add_argument('--is-dueling', default=False, type=bool)
    parser.add_argument('--is-noisynet', default=False, type=bool)
    parser.add_argument('--lr', default=0.001, type=float)  # 1e-3

    # distributional RL
    parser.add_argument('--v-min', default=-100, type=float)
    parser.add_argument('--v-max', default= 100, type=float)
    parser.add_argument('--atoms-length', default=51, type=int)

    param = parser.parse_args()

    return param

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.algo == 'td3':
        from algorithm.TD3.trainer import Trainer
        from algorithm.TD3.TD3 import TD3 as Agent
    elif args.algo == 'ppo':
        from algorithm.PPO.trainer import Trainer
        from algorithm.PPO.PPO import PPO as Agent
    elif args.algo == 'sac':
        from algorithm.SAC.sac_trainer import Trainer
        from algorithm.SAC.SAC import SAC as Agent
    elif args.algo=='dqn':
        #args.env_name = "CartPole-v1"
        args.env_name = "LunarLander-v3"
        from algorithm.DQN.trainer import Trainer
        print("is_Noisynet : ", args.is_noisynet)
        if args.is_noisynet:
            from algorithm.DQN.noisy_agent import NoisyDQN as Agent
        else:
            from algorithm.DQN.agent import DQN as Agent
    elif args.algo=='c51':
        args.env_name = "CartPole-v1"
        #args.env_name = "LunarLander-v3"
        from algorithm.Distributional_RL.trainer import Trainer
        from algorithm.Distributional_RL.C51 import C51Agent as Agent
    elif args.algo=='qrdqn':
        args.env_name = "CartPole-v1"
        #args.env_name = "LunarLander-v3"
        from algorithm.Distributional_RL.trainer import Trainer
        from algorithm.Distributional_RL.QRDQN import QR_DQN as Agent
    elif args.algo=='iqn':
        args.env_name = "CartPole-v1"
        #args.env_name = "LunarLander-v3"
        from algorithm.Distributional_RL.trainer import Trainer
        from algorithm.Distributional_RL.IQN import IQN as Agent

    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    agent = Agent(env,device,args)

    trainer = Trainer(env,eval_env, agent, args)
    trainer.run()


if __name__ == "__main__":
    args = get_parameters()
    main(args)