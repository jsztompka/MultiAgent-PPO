from model import PPONetwork
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
from utils import *
from torch.functional import F

from unityagents import UnityEnvironment

class EnvWrapper:
    def __init__(self, task):
        self.action_space = self.brain.vector_action_space_size

class CountScore:
    def __init__(self):
        self.total_episode = 100
        self.episode_rewards = np.zeros(self.total_episode)
        self.current_episode = 0

    def add_score(self, score):
        self.episode_rewards[self.current_episode] = score

        self.current_episode += 1
        self.current_episode = self.current_episode % 100


    def mean_score(self):
        return np.mean(self.episode_rewards)

class UnityTask:
    def __init__(self,  name):



        self.brain = None
        self.brain_name = None
        self.env = self.create_unity_env()

        #env details
        self.action_space = self.brain.vector_action_space_size
        self.observation_space = self.brain.vector_observation_space_size

        print(f'Action space {self.action_space}')
        print(f'State space {self.observation_space}')

        self.name = name

        #backwards compatibility
        self.action_dim = self.action_space
        #self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.observation_space))

        self.train_mode = True



    def extract_env_details(self,env_info):
        next_state = env_info.vector_observations  # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished

        return next_state, reward, done

    def create_unity_env(self):
        env = UnityEnvironment(file_name="Env\Tennis_Windows_x86_64\Tennis.exe")

        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]



        return env

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self.extract_env_details(env_info)[0]

    def step(self, actions):
        # beta distribution outputs actions between 0-1 and this converts them to -1,1 range
        actions = (actions - 0.5) * 2.0

        self.env.step(actions)[self.brain_name]


        env_info = self.env.step(actions)[self.brain_name]
        next_states, rewards, dones = self.extract_env_details(env_info)

        return next_states, rewards, np.array(dones)

        # return next_state, reward, np.array([done])

class PPOAgent_Unity():
    def __init__(self, config):

        self.config = config
        self.task = UnityTask('reacher')
        self.network = PPONetwork(self.config.state_dim, self.config.action_dim, 1000).to('cuda:0')
        self.opt = torch.optim.Adam(self.network.parameters(),  config.lr, amsgrad= True)
        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []
        self.states = self.task.reset()
        self.state_normalizer = None


        self.min_lr = self.config.lr * 0.3

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, gamma=0.8, step_size=200)
        self.max_score = 0
        self.episode_score = CountScore()



        if config.play_only:
            self.load_model()


    def load_model(self):
        self.network.load_state_dict(torch.load(self.config.saved_checkpoint))
        self.network.to('cuda:0')


    def update_lr(self):

        if self.total_steps < 1000:
            return

        if self.total_steps % 40000:
            # gradient clip
            # if self.config.gradient_clip > 0.1:
            # self.config.gradient_clip = 0.1

            if self.config.ppo_ratio_clip > 0.1:
                self.config.ppo_ratio_clip = 0.07


        #     # self.config.lr *= 0.5
        #     # self.opt = self.config.optimizer_fn(self.network.parameters(), self.config.lr)
        #     self.min_lr = self.config.lr * 0.2

        if self.total_steps % 30000 == 0 and self.config.entropy_weight > 0:
            self.config.entropy_weight -= 0.04
            if self.config.entropy_weight < 0:
                self.config.entropy_weight = 0.0





    def build_trajectory(self,memory_buffer):

        states = self.states
        episode = 0

        for _ in range(self.config.rollout_length):
            states = tensor(states)
            prediction = self.network(states)
            next_states, rewards, terminals = self.task.step((prediction['a'].cpu().numpy()))

            self.online_rewards += rewards

            if np.any(terminals):
                self.episode_score.add_score(np.max(self.online_rewards))
                self.score_data.append(self.episode_score.mean_score())

                self.episode_rewards.append(self.online_rewards[-1])
                self.online_rewards[:] = 0


            memory_buffer.add(prediction)
            memory_buffer.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)
                         })
            states = next_states

        current_score = self.episode_score.mean_score()
        print(f'Ep={episode}s current score {current_score} online rewards {self.online_rewards.mean()}')
        if self.max_score < self.online_rewards.mean():
            # torch.save(self.network.state_dict(), 'checkpoints\PPO-Tennis-Beta{}.pth'.format(int(self.total_steps)))
            self.max_score = current_score

        return memory_buffer


    def step(self):

        self.update_lr()

        if self.config.play_only:
            self.validate(False)
            return

        config = self.config
        memory_buffer = Storage(config.rollout_length)
        states = self.states

        self.network.eval()

        #save trajectories into memory buffer - no training here all episodes are recorded using online policy
        memory_buffer = self.build_trajectory(memory_buffer)
       
        # current score is calcualted for statistics only
        current_score = self.online_rewards.mean()
        print(f'Current score {current_score}')

        if not config.play_only:

            # Save checkpoint if model has improved
            if self.max_score < current_score and not config.play_only:
                torch.save(self.network, '/checkpoint/PPO-{}.pth'.format(int(current_score)))
                self.max_score = current_score

            self.states = states
            states = tensor(states)
            prediction = self.network(states)
            memory_buffer.add(prediction)
            memory_buffer.placeholder()

            advantages = tensor(np.zeros((config.num_workers, 1)))
            returns = prediction['v'].detach()
            for i in reversed(range(config.rollout_length)):
                returns = memory_buffer.r[i] + config.discount * memory_buffer.m[i] * returns

                # GAE
                td_error = memory_buffer.r[i] + config.discount * memory_buffer.m[i] * memory_buffer.v[i + 1] - memory_buffer.v[i]
                advantages = advantages * config.gae_tau * config.discount * memory_buffer.m[i] + td_error

                memory_buffer.adv[i] = advantages.detach()
                memory_buffer.ret[i] = returns.detach()

            # using pre-recorded trajectories train agent
            batch_steps = self.train_agent(memory_buffer)



            steps = batch_steps
            # * config.num_workers
            self.total_steps += steps
            self.lr_scheduler.step()

            if self.total_steps % 50000 == 0:
                self.validate(False)

        else:
            self.validate(False)

    def train_agent(self, memory_buffer):

        states, actions, log_probs_old, returns, advantages = memory_buffer.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        sum_returns = 0
        sum_advantage = 0
        sum_policy_loss = 0
        sum_critic_loss = 0
        sum_entropy = 0
        batch_steps = 0

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=config.optimization_epochs, eta_min=self.min_lr)

        self.network.train()

        config = self.config

        for ep in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)

            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                #this activates only part of the network responsible for V and log_policy
                #actions in this case are already provided and won't be calculated!
                prediction = self.network(sampled_states.cuda(), sampled_actions.cuda())

                #ratio is a diff between old and newly calcualted policy
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                obj = ratio * sampled_advantages

                # gradient clip (1 - epsilon / 1 + epsilon happens here)
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages

                # entropy_weight is a factor for entropy boost - it should be set to 0 once the training stabilises
                policy_loss = torch.min(obj, obj_clipped).mean() + config.entropy_weight * prediction['ent'].mean()


                # Huber loss
                value_loss = F.smooth_l1_loss(prediction['v'], sampled_returns.view(-1, 1))

                sum_returns, sum_advantage, sum_policy_loss, sum_critic_loss, sum_entropy = \
                    self.log_stats(sampled_returns, sampled_advantages, policy_loss, value_loss,
                                   prediction['ent'].mean(),
                                   batch_steps, sum_returns, sum_advantage, sum_critic_loss, sum_policy_loss,
                                   sum_entropy)
                batch_steps += 1

                self.opt.zero_grad()
                (-(policy_loss - value_loss)).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

            # lr_scheduler.step()

        return batch_steps


    def get_lr(self):
        for param_group in self.opt.param_groups:
            return param_group['lr']

    def log_stats(self, returns, advantage, loss, critic_loss, entropy, batch_step, sum_returns, sum_advantage, sum_critic, sum_loss, sum_entropy):
        # track statistics

        sum_returns += returns.mean()
        sum_advantage += advantage.mean()
        #sum_loss_actor += act_loss
        sum_critic += critic_loss
        sum_loss += loss
        sum_entropy += entropy.mean()



        logger = self.config.logger


        frame_idx = self.total_steps # / self.config.rollout_length

        batch_count = self.config.optimization_epochs * (self.config.rollout_length / self.config.mini_batch_size)

        step_idx = batch_step + frame_idx

        batch_step += 1


        logger.add_scalar("returns", sum_returns / batch_step, step_idx)
        logger.add_scalar("advantage", sum_advantage / batch_step, step_idx)
        #logger.add_scalar("loss_actor", sum_loss_actor / batch_count, frame_idx)
        logger.add_scalar("loss_critic", sum_critic / batch_step, step_idx)
        logger.add_scalar("entropy", sum_entropy / batch_step, step_idx)
        logger.add_scalar("loss_total", sum_loss / batch_step, step_idx)
        logger.add_scalar("lr", self.get_lr(), step_idx)

        return sum_returns, sum_advantage, sum_loss, sum_critic, sum_entropy

    def validate(self, fast_test=True):
        score = np.zeros(self.config.num_workers)
        self.network.eval()

        self.task.train_mode = fast_test





        actual_score = 0
        for i in range(10):
            print(f"Testing {i} score={np.mean(score)}")
            terminals = np.zeros(2)
            states = self.task.reset()
            ep_scores = []
            while not all(terminals):
                states = tensor(states)
                prediction = self.network(states)
                next_states, rewards, terminals = self.task.step((prediction['a']).cpu().numpy())
                score += rewards
                states = next_states

            # score.append(np.mean(ep_scores))


            # 100 episodes takes too long
            self.task.train_mode = False
        actual_score = np.mean(score)
        print(f'Ep: 100 {actual_score}')


