import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
import gym
import math
import time
import datetime

#just a wrapper to get observation dimension and action dimension
class GymWrapper():
    def __init__(self, env):
        self.env = env        

    def is_discrete(self):
        return type(self.env.action_space) is not gym.spaces.Box

    def get_obs_dim(self):
        if type(self.env.observation_space) is gym.spaces.Box:
            state_dim = 1

            if len(self.env.observation_space.shape) > 1:                
                for i in range(len(self.env.observation_space.shape)):
                    state_dim *= self.env.observation_space.shape[i]            
            else:
                state_dim = self.env.observation_space.shape[0]

            return state_dim
        else:
            return self.env.observation_space.n
            
    def get_action_dim(self):
        if self.is_discrete():
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu = True):
        super(Policy_Model, self).__init__()

        self.temperature = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.alpha = nn.parameter.Parameter(
          torch.Tensor([1.0])
        )

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU()
        )

        self.actor_layer = nn.Sequential(
          nn.Linear(64, action_dim),
          nn.Softmax(-1)
        )
        
    def forward(self, states, detach = False):
      x = self.nn_layer(states)

      if detach:
        return self.actor_layer(x).detach(), self.temperature.detach(), self.alpha.detach()
      else:
        return self.actor_layer(x), self.temperature, self.alpha
      
class Value_Model(nn.Module):
    def __init__(self, state_dim, use_gpu = True):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
          nn.Linear(state_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
        )
        
    def forward(self, states, detach = False):
      if detach:
        return self.nn_layer(states).detach()
      else:
        return self.nn_layer(states)

class BasicDiscrete():
    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().int()
        
    def entropy(self, datas):
        distribution = Categorical(datas)
        return distribution.entropy().unsqueeze(1)
        
    def logprob(self, datas, value_data):
        distribution = Categorical(datas)        
        return distribution.log_prob(value_data).unsqueeze(1)

    def kldivergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)
        return kl_divergence(distribution1, distribution2).unsqueeze(1)

    def deterministic(self, datas):
        return int(torch.argmax(datas, 1))

class PolicyMemory(Dataset):
    def __init__(self, capacity = 100000, datas = None):
        self.capacity       = capacity
        self.position       = 0

        if datas is None:
            self.states         = []
            self.actions        = []
            self.rewards        = []
            self.dones          = []
            self.next_states    = []
        else:
            self.states, self.actions, self.rewards, self.dones, self.next_states = datas
            if len(self.dones) >= self.capacity:
                raise Exception('datas cannot be longer than capacity')        

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx], dtype = torch.float32), torch.tensor(self.actions[idx], dtype = torch.float32), \
            torch.tensor([self.rewards[idx]], dtype = torch.float32), torch.tensor([self.dones[idx]], dtype = torch.float32), \
            torch.tensor(self.next_states[idx], dtype = torch.float32)

    def save_obs(self, state, action, reward, done, next_state):
        if len(self) >= self.capacity:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.next_states[0]

        self.states.append(deepcopy(state))
        self.actions.append(deepcopy(action))
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(deepcopy(next_state))

    def save_replace_all(self, states, actions, rewards, dones, next_states):
        self.clear_memory()
        self.save_all(states, actions, rewards, dones, next_states)

    def save_all(self, states, actions, rewards, dones, next_states):
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.save_obs(state, action, reward, done, next_state)

    def get_all_items(self):         
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def get_ranged_items(self, start_position = 0, end_position = None):   
        if end_position is not None and end_position != -1:
            states      = self.states[start_position:end_position + 1]
            actions     = self.actions[start_position:end_position + 1]
            rewards     = self.rewards[start_position:end_position + 1]
            dones       = self.dones[start_position:end_position + 1]
            next_states = self.next_states[start_position:end_position + 1]
        else:
            states      = self.states[start_position:]
            actions     = self.actions[start_position:]
            rewards     = self.rewards[start_position:]
            dones       = self.dones[start_position:]
            next_states = self.next_states[start_position:]

        return states, actions, rewards, dones, next_states 

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]

    def clear_idx(self, idx):
        del self.states[idx]
        del self.actions[idx]
        del self.rewards[idx]
        del self.dones[idx]
        del self.next_states[idx]

class GeneralizedAdvantageEstimation():
    def __init__(self, gamma = 0.99):
        self.gamma  = gamma

    def compute_advantages(self, rewards, values, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * (1.0 - self.gamma) * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)

class ValueLoss():
    def __init__(self, advantage_function, value_clip):
        self.advantage_function = advantage_function
        self.value_clip = value_clip

    def compute_loss(self, values, old_values, next_values, rewards, dones):
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones)
        returns     = (advantages + values).detach()

        if self.value_clip is None:
            value_loss      = ((returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped    = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip)
            value_loss      = ((returns - vpredclipped).pow(2) * 0.5).mean()

        return value_loss

class AlphaLoss():
    def __init__(self, distribution, coef_alpha_upper = torch.Tensor([0.01]), coef_alpha_below = torch.Tensor([0.005])):
        self.distribution       = distribution

        self.coef_alpha_upper  = coef_alpha_upper
        self.coef_alpha_below  = coef_alpha_below

    def compute_loss(self, action_datas, old_action_datas, alpha):
        coef_alpha  = torch.distributions.Uniform(self.coef_alpha_below.log(), self.coef_alpha_upper.log()).sample().exp()
        Kl          = self.distribution.kldivergence(old_action_datas, action_datas)
       
        loss        = alpha * (coef_alpha - Kl.squeeze().detach()) + alpha.detach() * Kl.squeeze()
        return loss.mean()

class TemperatureLoss():
    def __init__(self, advantage_function, coef_temp = 0.0001, device = torch.device('cuda')):
        self.advantage_function = advantage_function
        self.coef_temp          = coef_temp
        self.device             = device

    def compute_loss(self, values, next_values, rewards, dones, temperature):
        advantages  = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()                
        top_adv, _  = torch.topk(advantages, math.ceil(advantages.size(0) / 2), 0)

        n           = torch.Tensor([top_adv.size(0)]).to(self.device)
        ratio       = top_adv / (temperature + 1e-3)

        loss        = temperature * self.coef_temp + temperature * (torch.logsumexp(ratio, dim = 0) - n.log())
        return loss.squeeze()

class PhiLoss():
    def __init__(self, distribution, advantage_function):
        self.advantage_function = advantage_function
        self.distribution       = distribution

    def compute_loss(self, action_datas, values, next_values, actions, rewards, dones, temperature):
        temperature         = temperature.detach()

        advantages          = self.advantage_function.compute_advantages(rewards, values, next_values, dones).detach()
        top_adv, top_idx    = torch.topk(advantages, math.ceil(advantages.size(0) / 2), 0)

        logprobs            = self.distribution.logprob(action_datas, actions)
        top_logprobs        = logprobs[top_idx]        

        ratio               = top_adv / (temperature + 1e-3)
        psi                 = torch.nn.functional.softmax(ratio, dim = 0)

        loss                = -1 * (psi * top_logprobs).sum()
        return loss

class EntropyLoss():
    def __init__(self, distribution, entropy_coef = 0.1):
        self.distribution       = distribution
        self.entropy_coef       = entropy_coef

    def compute_loss(self, action_datas):
        loss                = -1 * self.entropy_coef * self.distribution.entropy(action_datas).mean()

        return loss

class AgentVMPO():  
    def __init__(self, policy, value, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss,
            policy_memory, policy_optimizer, value_optimizer, policy_epochs = 1, is_training_mode = True, batch_size = 32, folder = 'model', 
            device = torch.device('cuda:0'), old_policy = None, old_value = None):   

        self.batch_size         = batch_size  
        self.policy_epochs      = policy_epochs
        self.is_training_mode   = is_training_mode
        self.folder             = folder

        self.policy             = policy
        self.old_policy         = old_policy
        self.value              = value
        self.old_value          = old_value

        self.distribution       = distribution
        self.policy_memory      = policy_memory
        
        self.alpha_loss         = alpha_loss
        self.phi_loss           = phi_loss
        self.temperature_loss   = temperature_loss
        self.value_loss         = value_loss
        self.entropy_loss       = entropy_loss        

        self.policy_optimizer   = policy_optimizer
        self.value_optimizer    = value_optimizer   
        self.device             = device

        self.i_update           = 0

        if self.old_policy is None:
            self.old_policy  = deepcopy(self.policy)

        if self.old_value is None:
            self.old_value  = deepcopy(self.value)

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()
        
    @property
    def memory(self):
        return self.policy_memory

    def _training(self, states, actions, rewards, dones, next_states):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        action_datas, temperature, alpha    = self.policy(states)
        old_action_datas, _, _              = self.old_policy(states, True)       
        values                              = self.value(states)
        old_values                          = self.old_value(states, True)
        next_values                         = self.value(next_states, True)        
        
        phi_loss    = self.phi_loss.compute_loss(action_datas, values, next_values, actions, rewards, dones, temperature)
        temp_loss   = self.temperature_loss.compute_loss(values, next_values, rewards, dones, temperature)
        alpha_loss  = self.alpha_loss.compute_loss(action_datas, old_action_datas, alpha)
        value_loss  = self.value_loss.compute_loss(values, old_values, next_values, rewards, dones)
        ent_loss    = self.entropy_loss.compute_loss(action_datas)

        loss    = phi_loss + temp_loss + alpha_loss + value_loss + ent_loss
        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def update(self):
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_value.load_state_dict(self.value.state_dict())

        for _ in range(self.policy_epochs):
            dataloader = DataLoader(self.policy_memory, self.batch_size, shuffle = False)
            for states, actions, rewards, dones, next_states in dataloader:
                self._training(states.float().to(self.device), actions.float().to(self.device), rewards.float().to(self.device), dones.float().to(self.device), next_states.float().to(self.device))

        self.policy_memory.clear_memory()

    def act(self, state):
        with torch.inference_mode():
            state               = torch.FloatTensor(state).unsqueeze(0).float().to(self.device)
            action_datas, _, _  = self.policy(state)
            
            if self.is_training_mode:
                action = self.distribution.sample(action_datas)
            else:
                action = self.distribution.deterministic(action_datas)

            action = action.squeeze(0).detach().tolist()
              
        return action

    def logprobs(self, state, action):
        with torch.inference_mode():
            state               = torch.FloatTensor(state).unsqueeze(0).float().to(self.device)
            action_datas, _, _  = self.policy(state)
            
            logprobs        = self.distribution.logprob(action_datas, action)
            logprobs        = logprobs.squeeze(0).detach().tolist()

        return logprobs

    def save_obs(self, state, action, reward, done, next_state):
        self.policy_memory.save_obs(state, action, reward, done, next_state)

    def save_weights(self, folder = None):
        if folder == None:
            folder = self.folder
            
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, self.folder + '/v_mpo.pth')
        
    def load_weights(self, folder = None, device = None):
        if device is None:
            device = self.device

        if folder is None:
            folder = self.folder

        model_checkpoint = torch.load(self.folder + '/v_mpo.pth', map_location = device)
        self.policy.load_state_dict(model_checkpoint['policy_state_dict'])        
        self.value.load_state_dict(model_checkpoint['value_state_dict'])
        
        if self.policy_optimizer is not None:
            self.policy_optimizer.load_state_dict(model_checkpoint['policy_optimizer_state_dict'])

        if self.value_optimizer is not None:
            self.value_optimizer.load_state_dict(model_checkpoint['value_optimizer_state_dict'])

        if self.is_training_mode:
            self.policy.train()
            self.value.train()

        else:
            self.policy.eval()
            self.value.eval()

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def set_weights(self, policy_weights, value_weights):
        self.policy.load_state_dict(policy_weights)
        self.value.load_state_dict(value_weights)

class IterRunner():
    def __init__(self, agent, env, is_save_memory, render, n_update, is_discrete, max_action, writer = None, n_plot_batch = 100):
        self.agent              = agent
        self.env                = env

        self.render             = render
        self.is_save_memory     = is_save_memory
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch
        self.is_discrete        = is_discrete

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = self.env.reset()

    def run(self):
        for _ in range(self.n_update):
            action                      = self.agent.act(self.states)
            next_state, reward, done, _ = self.env.step(action)
            
            if self.is_save_memory:
                self.agent.save_obs(self.states.tolist(), action, reward, float(done), next_state.tolist())
                
            self.states         = next_state
            self.eps_time       += 1 
            self.total_reward   += reward
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0    

        return self.agent.memory.get_ranged_items(-self.n_update)

class Executor():
    def __init__(self, agent, n_iteration, runner, save_weights = False, n_saved = 10, load_weights = False, is_training_mode = True):
        self.agent              = agent
        self.runner             = runner

        self.n_iteration        = n_iteration
        self.save_weights       = save_weights
        self.n_saved            = n_saved
        self.is_training_mode   = is_training_mode 
        self.load_weights       = load_weights       

    def execute(self):
        if self.load_weights:
            self.agent.load_weights()
            print('Weight Loaded')  

        start = time.time()
        print('Running the training!!')

        try:
            for i_iteration in range(1, self.n_iteration, 1):
                self.runner.run()                

                if self.is_training_mode:
                    self.agent.update()

                    if self.save_weights:
                        if i_iteration % self.n_saved == 0:
                            self.agent.save_weights()
                            print('weights saved')

        except KeyboardInterrupt:
            print('Stopped by User')
        finally:
            finish = time.time()
            timedelta = finish - start
            print('\nTimelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

############## Hyperparameters ##############

load_weights            = False # If you want to load the agent, set this to True
save_weights            = False # If you want to save the agent, set this to True
is_training_mode        = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
render                  = True # If you want to display the image. Turn this off if you run this in Google Collab
reward_threshold        = 1000 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

n_plot_batch            = 1 # How many episode you want to plot the result
n_iteration             = 100000000 # How many episode you want to run
n_update                = 128 # How many episode before you update the Policy
n_saved                 = 1

coef_alpha_upper        = 0.01
coef_alpha_below        = 0.005

coef_temp               = 0.01
batch_size              = 64
policy_epochs           = 5
value_clip              = None
action_std              = 1.0
gamma                   = 0.95
learning_rate           = 1e-4
entropy_coef            = 0.1

device                  = torch.device('cuda')
folder                  = 'weights'

env                     = gym.make('CartPole-v1')

state_dim               = None
action_dim              = None
max_action              = 1

#####################################################################################################################################################
environment         = GymWrapper(env)

if state_dim is None:
    state_dim = environment.get_obs_dim()
print('state_dim: ', state_dim)

if environment.is_discrete():
    print('discrete')
else:
    print('continous')

if action_dim is None:
    action_dim = environment.get_action_dim()
print('action_dim: ', action_dim)

coef_alpha_upper    = torch.Tensor([coef_alpha_upper]).to(device)
coef_alpha_below    = torch.Tensor([coef_alpha_below]).to(device)

distribution        = BasicDiscrete()
advantage_function  = GeneralizedAdvantageEstimation(gamma)
policy_memory       = PolicyMemory()

alpha_loss          = AlphaLoss(distribution, coef_alpha_upper, coef_alpha_below)
phi_loss            = PhiLoss(distribution, advantage_function)
temperature_loss    = TemperatureLoss(advantage_function, coef_temp, device)
value_loss          = ValueLoss(advantage_function, value_clip)
entropy_loss        = EntropyLoss(distribution, entropy_coef)

policy              = Policy_Model(state_dim, action_dim).float().to(device)
value               = Value_Model(state_dim).float().to(device)
policy_optimizer    = AdamW(policy.parameters(), lr = learning_rate)
value_optimizer     = AdamW(value.parameters(), lr = learning_rate)

agent   = AgentVMPO(policy, value, distribution, alpha_loss, phi_loss, entropy_loss, temperature_loss, value_loss,
            policy_memory, policy_optimizer, value_optimizer, policy_epochs, is_training_mode, batch_size, folder, 
            device)

runner      = IterRunner(agent, environment, is_training_mode, render, n_update, environment.is_discrete, max_action, SummaryWriter(), n_plot_batch)
executor    = Executor(agent, n_iteration, runner, save_weights, n_saved, load_weights, is_training_mode)

executor.execute()