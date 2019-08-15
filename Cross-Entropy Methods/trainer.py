import torch
import torch.optim as optim



class Trainer:

    def __init__(self, optimizer, function_approximator, target_function_approximator, loss_fn, gamma=0.95):
        self.optimizer = optimizer
        self.function_approximator = function_approximator
        self.target_function_approximator = target_function_approximator
        self.loss_fn = loss_fn
        self.gamma = gamma

    def train(self, batch):
        states, actions, rewards, dones, next_states = self._get_torch_tensors(batch)
        calculated_q_vals = self.function_approximator(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_state_values = self.target_function_approximator(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        target_q_vals = rewards + self.gamma * next_state_values.detach()

        loss = self.loss_fn(calculated_q_vals, target_q_vals)
        loss.backward()
        self.optimizer.step()


    def _get_torch_tensors(self, batch):
        states, actions, rewards, dones, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.ByteTensor(dones)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        return states, actions, rewards, dones, next_states
