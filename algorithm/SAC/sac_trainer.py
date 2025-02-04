import numpy as np

class Trainer:
    def __init__(self, env, eval_env, agent, args):
        self.args = args

        self.agent = agent
        self.env = env
        self.eval_env = eval_env

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

#        self.eval_flag = args.eval_flag
#        self.eval_episode = args.eval_episode
#        self.eval_freq = args.eval_freq
#        self.checkpoint_freq = args.checkpoint_freq

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.finish_flag = False

    def run(self):

        while not self.finish_flag:

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            state,_  = self.env.reset()
            done = False

            while not done:
                self.total_step += 1
                self.local_step += 1

                if self.total_step >= self.start_step:
                    action = self.agent.get_action(state, evaluation=False)
                else:
                    # exploration : Gaussian Noise
                    action = self.agent.get_action(state, evaluation=True)

                # 위에서 정해진 액션으로 환경과 상호작용.
                next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
                done = terminated or truncated
                self.episode_reward += reward

                self.agent.buffer.push(state, action, reward, next_state, done)

                state = next_state

                # Update parameters
                # replay buffer에 경험이 쌓이고, total_step이 upate 해야 할 step에 도달한 후, 업데이트 주기에 맞춰 학습 시작
                if self.agent.buffer.size >= self.batch_size and self.total_step >= self.update_after and self.total_step % self.update_every == 0:
                    total_actor_loss = 0
                    total_critic_loss = 0
                    total_log_alpha_loss = 0
                    for _ in range(self.update_every):
                        # agent.train으로 loss function 누적
                        # update_every 와 학습 주기를 1대1로 하기 위함
                        critic_loss, actor_loss, log_alpha_loss = self.agent.train(self.args)
                        total_critic_loss += critic_loss
                        total_actor_loss += actor_loss
                        total_log_alpha_loss += log_alpha_loss

                    # Print loss.
                    if self.args.show_loss:
                        print("Loss  |  Actor loss {:.3f}  |  Critic loss {:.3f}  |  Log-alpha loss {:.3f}"
                              .format(total_actor_loss / self.update_every, total_critic_loss / self.update_every,
                                      total_log_alpha_loss / self.update_every))
                # Evaluation.
                # eval_flag = 1 일 때 eva_flag 마다 평가.
                # 100 번째 줄 코드와 차이가 evaluate는 평가만 하는 건가?
                # Raise finish_flag.
                if done:
                    print(f'Episode {self.episode} | reward : {self.episode_reward} | total_step : {self.total_step}')
                if self.total_step == self.max_step:
                    self.finish_flag = True