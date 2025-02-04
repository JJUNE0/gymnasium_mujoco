import torch
import numpy as np

class Trainer:
    def __init__(self, env, eval_env, agent, args):

        self.args =args

        self.env = env
        self.eval_env = eval_env
        self.agent = agent

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every
        self.max_episode = args.max_episode

#        self.eval_flag = args.eval_flag
#        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq
        self.checkpoint_freq = args.checkpoint_freq

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.finish_flag = False

        self.epochs = 20
        self.score = 0

    def eval_policy(self,  eval_episodes=10):
        #eval_env = gym.make(env_name)
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.agent.get_action(state, evaluation=True)
                state, reward, terminated, truncated ,_ = self.eval_env.step(action)
                done = terminated or truncated
                avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def run(self):

        while not self.finish_flag:
            self.episode+=1
            self.episode_reward = 0
            self.local_step = 0

            state,_ = self.env.reset()
            done = False

            while not done:
                self.total_step+=1
                self.local_step+=1

                if self.total_step >= self.start_step:
                    action = self.agent.get_action(state, evaluation=True)
                else:
                    # exploration : Gaussian Noise
                    action = self.agent.get_action(state, evaluation=False)

                #print(action)
                next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
                done = terminated or truncated

                self.agent.store_sample(state, action, reward, next_state, terminated)
                self.episode_reward += reward
                self.score += reward
                state = next_state

                # replay buffer에 경험이 쌓이고, total_step이 upate 해야 할 step에 도달한 후, 업데이트 주기에 맞춰 학습 시작
                if self.agent.buffer.size >= self.batch_size and self.total_step >= self.update_after and self.total_step % self.update_every == 0:
                    for _ in range(self.update_every):
                        self.agent.train()

                    # Update target networks at specified frequency
                    #if self.total_step % self.update_every == 0:
                if done:
                    print(f'Episode {self.episode} | reward : {self.episode_reward} | total_step : {self.total_step}')

                if (self.total_step+1) % self.eval_freq == 0:
                    self.eval_policy()


                if self.total_step == self.max_step or self.episode == self.max_episode:

                    print(f'Reach Max step {self.max_step}, Average episode reward {self.score/self.episode}')
                    torch.save(self.agent.actor.state_dict(), "checkpoint/td3/actor_parameters.pth")
                    torch.save(self.agent.actor_target.state_dict(), "checkpoint/td3/actor_target_parameters.pth")
                    torch.save(self.agent.critic.state_dict(), "checkpoint/td3/critic_parameters.pth")
                    #torch.save(self.agent.critic2.state_dict(), "checkpoint/td3/critic2_parameters.pth")
                    torch.save(self.agent.critic_target.state_dict(), "checkpoint/td3/critic_target_parameters.pth")
                    #torch.save(self.agent.critic_target2.state_dict(), "checkpoint/td3/critic_target2_parameters.pth")

                    self.finish_flag = True

                    break

