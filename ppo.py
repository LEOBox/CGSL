from osim.env import ProstheticsEnv
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import pickle

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(Critic, self).__init__()
        
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        self.block = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
        )
        
    def forward(self, state):
        out = self.block(state)
        return out

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        self.block_state = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            #nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        
        self.block_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            #nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        
        self.block_mean = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )
        
        self.log_std = nn.Parameter(torch.ones(action_size))
        
        #self.block_std = nn.Sequential(
        #    nn.Linear(hidden_size, action_size),
        #)
        
    def forward(self, state):
        out = self.block_state(state)
        out = self.block_hidden(out)
        mean = self.block_mean(out)
        std = torch.ex
        return mean ,std


class PPO(object):
    
    def __init__(self, actor, critic, actor_opt, critic_opt,
                 update_epoch=30, thread=10, gamma=0.95, tau=0.95, punish=0.1,
                 actor_grad_norm=0.5, critic_grad_norm=0.5):
        super(PPO, self).__init__()
        self.actor = actor
        self.critic = critic
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.update_epoch = update_epoch
        self.thread = thread
        self.gamma = gamma
        self.tau = tau
        self.punish = punish
        self.actor_grad_norm = actor_grad_norm
        self.critic_grad_norm = critic_grad_norm

    def select_action(self, state):
        state = torch.tensor(state,dtype=torch.float).unsqueeze(0)
        mean,std = self.actor(state)
        dist = Normal(mean, torch.exp(std))
        action = dist.sample()
        #action += torch.randn(1,19) * 0.1
        log_prob = dist.log_prob(action)
        action = torch.clamp(action,0,1)
        return action.squeeze(0).detach().numpy(), log_prob.detach().numpy()
    
    def select_action_evaluate(self, state):
        state = torch.tensor(state,dtype=torch.float).unsqueeze(0)
        mean,std = self.actor(state)
        dist = Normal(mean, torch.exp(std))
        action = dist.sample()
        action = torch.clamp(action,0,1)
        return action.squeeze(0).detach().numpy()

    def evaluate_actions(self, state, action):
        mean,std = self.actor(state)
        dist = Normal(mean, torch.exp(std))
        log_prob = dist.log_prob(action)
        return log_prob

    def start_sim(self):
        torch.manual_seed(random.randint(1,100))
        env = ProstheticsEnv(visualize=False)
        memory = []
        observation = env.reset()
        for t in range(1000):
            state = observation
            action,log_prob = self.select_action(state)
            observation,reward,done,_ = env.step(action)
            e = [state,action,reward,log_prob]
            memory.append(e)
            if done:
                for i in range(len(memory)):
                    memory[i][2] -= self.punish
                break   

        return memory

    def start_sim_evaluate(self):
        env = ProstheticsEnv(visualize=False)
        total_reward = 0
        observation = env.reset()
        for t in range(1000):
            state = observation
            action = self.select_action_evaluate(state)
            observation,reward,done,_ = env.step(action)
            total_reward += reward
            if done:
                break   

        return total_reward

    def compute_advantage(self,rewards, values):
        discount_rewards = torch.zeros_like(rewards)
        deltas = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        pre_discount_reward = 0
        pre_value = 0
        pre_advantage = 0

        for i in reversed(range(rewards.size(0))):
            discount_rewards[i] = rewards[i] + self.gamma * pre_discount_reward
            deltas[i] = rewards[i] + self.gamma * pre_value - values[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * pre_advantage

            pre_discount_reward = discount_rewards[i]
            pre_value = values[i]
            pre_advantage = advantages[i]

        discount_rewards = (discount_rewards - discount_rewards.mean()) / (discount_rewards.std() + 1e-8)    
        advantages = (advantages - advantages.mean()) / advantages.std()

        return discount_rewards, advantages

    def update(self, memories):
        policy_loss_log = []
        value_loss_log = []
        states = torch.from_numpy(np.vstack(memory[0] for memory in memories)).float()
        actions = torch.from_numpy(np.vstack(memory[1] for memory in memories)).float()
        rewards = torch.from_numpy(np.vstack(memory[2] for memory in memories)).float()
        log_probs_hat = torch.from_numpy(np.vstack(memory[3] for memory in memories)).float()
        values = self.critic(states).data
        discount_rewards,advantages = self.compute_advantage(rewards, values)
        self.actor.train()
        self.critic.train()
        for _ in range(self.update_epoch):
            log_probs = self.evaluate_actions(states, actions)
            ratios = (log_probs - log_probs_hat).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-0.2, 1+0.2) * advantages
            policy_loss = -torch.min(surr1,surr2).mean()
            self.actor_opt.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_norm)
            self.actor_opt.step()
            values = self.critic(states)
            value_loss = (0.5 * (discount_rewards - values).pow(2)).mean()
            self.critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_norm)
            self.critic_opt.step()
            
            policy_loss_log.append(policy_loss.item())
            value_loss_log.append(value_loss.item())
            
        return policy_loss_log, value_loss_log

    def sample(self):
        results = []
        self.actor.eval()
        self.critic.eval()
        with mp.Pool() as pool:
            for _ in range(self.thread):
                result = pool.apply_async(self.start_sim,args=())
                results.append(result)
            memories = []
            for r in results:
                memories.extend(r.get())

        return memories

    def evaluate_policy(self):
        results = []
        self.actor.eval()
        self.critic.eval()
        with mp.Pool() as pool:
            for _ in range(self.thread):
                result = pool.apply_async(self.start_sim_evaluate,args=())
                results.append(result)
            rewards = [r.get() for r in results]

        return max(rewards), sum(rewards) / self.thread, min(rewards)
    
    def train(self, epoches, log_epoch):
        log_rewards = []
        log_policy_loss = []
        log_value_loss = []
        for i in range(epoches):
            memories = self.sample()
            policy_loss_log,value_loss_log = self.update(memories)
            log_policy_loss.extend(policy_loss_log)
            log_value_loss.extend(value_loss_log)
            if i % log_epoch == 0:
                reward_max, reward_mean, reward_min = self.evaluate_policy()
                log_rewards.append([reward_max,reward_mean,reward_min])
                print("Epoch {}: Max - {}; Mean - {}; Min - {}".format(str(i+1),
                                                                      str(reward_max),
                                                                      str(reward_mean),
                                                                      str(reward_min)))
        reward_max, reward_mean, reward_min = self.evaluate_policy()
        log_rewards.append([reward_max,reward_mean,reward_min])
        print("Epoch {}: Max - {}; Mean - {}; Min - {}".format(str(epoches+1),
                                                              str(reward_max),
                                                              str(reward_mean),
                                                              str(reward_min)))
                
        return log_rewards, log_policy_loss, log_value_loss



