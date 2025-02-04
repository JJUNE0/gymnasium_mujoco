
import torch
import matplotlib
import matplotlib.pyplot as plt

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
#        self.eval_freq = args.eval_freq
        self.checkpoint_freq = args.checkpoint_freq

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.finish_flag = False

        self.epochs = 20
        self.score = 0

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

                mu_old, std_old, action = self.agent.get_action(state)
                """ 나중에 수식 이해 """
                log_old_policy = self.agent.log_pdf(mu_old, std_old, action)

                next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
                done = terminated or truncated
                # ppo 용 sample 저장.
                # print(log_old_policy)

                self.agent.store_sample(state, action, reward, next_state, log_old_policy, done)
                # replay buffer에 경험이 쌓이고, total_step이 upate 해야 할 step에 도달한 후, 업데이트 주기에 맞춰 학습 시작

                #print(self.agent.buffer.size)
                if self.agent.buffer.size == self.agent.buffer.capacity:
                    self.agent.train()

                self.episode_reward += reward
                self.score += reward
                state = next_state

                if done:
                    print(f'Episode {self.episode} | reward : {self.episode_reward} | total_step : {self.total_step}')

                if self.total_step == self.max_step or self.episode == self.max_episode:
                    print(f'Reach Max step {self.max_step}, Average episode reward {self.score/self.episode}')
                    torch.save(self.agent.actor.state_dict(), "checkpoint/ppo/actor_parameters.pth")
                    torch.save(self.agent.critic.state_dict(), "checkpoint/ppo/critic_parameters.pth")

                    self.finish_flag = True
