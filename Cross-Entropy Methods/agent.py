from rl_utils import get_action
from experience_replay import experience

class Agent:

    def __init__(self, env, experience_buffer, function_approximator):
        self.env = env
        self.experience_buffer = experience_buffer
        self.function_approximator = function_approximator

    def run_episode(self, trainer, eps, gamma, lambda_=1, batch_size=32, min_buffer_size=100):
        episode_reward = 0
        state = self.env.reset()
        trajectory = []
        done = False

        while not done:
            action = get_action(state, self.env, self.function_approximator, eps)
            next_state, reward, done, info = env.step(action)
            exp = experience(state, action, reward, done, next_state)
            trajectory.append(exp)
            state = next_state
            episode_reward += reward

        if len(self.experience_buffer) > min_buffer_size:
            trainer.train( experience_buffer.sample(batch_size) )

        self._add_records_to_experience_buffer(trajectory, lambda_, gamma)
        return episode_reward

    def _add_records_to_experience_buffer(self, trajectory, lambda_, gamma):
        num_timesteps = len(trajectory)
        for timestep, exp in enumerate(trajectory):
            done = True if timestep + lambda_ >= num_timesteps else False
            next_state = exp.state if done else trajectory[timestep + lambda_].next_state
            reward = 0
            for exp_ in trajectory[ min(timestep+lambda_-1, num_timesteps): timestep-1: -1 ]:
                reward = reward * gamma + exp_.reward

            exp_lambda = experience(exp.state, exp.action, reward, done, next_state)
            self.experience_buffer.append(exp_lambda)



if __name__ == '__main__':

    import gym
    from experience_replay import ExperienceBuffer
    from function_approximators import DQN
    from trainer import Trainer
    import torch
    import torch.nn as nn
    import torch.optim as optim

    GAMMA = 0.99
    MAX_BUFFER_SIZE = 1000
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    BATCH_SIZE = 32
    MIN_BUFFER_SIZE = 100
    LAMBDA = 4

    eps = 1.0
    eps_decay = 0.99

    env = gym.make('CartPole-v0')
    experience_buffer = ExperienceBuffer(MAX_BUFFER_SIZE)
    function_approximator = DQN(env.observation_space.shape[0], env.action_space.n)
    target_function_approximator = DQN(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(function_approximator.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    trainer = Trainer(optimizer, function_approximator, target_function_approximator, loss_fn, gamma=GAMMA)

    agent = Agent(env, experience_buffer, function_approximator)

    NUM_EPISODES = 10000
    for episode in range(NUM_EPISODES):
        reward = agent.run_episode(trainer, eps=eps, gamma=GAMMA, lambda_=4, batch_size=BATCH_SIZE, min_buffer_size=MIN_BUFFER_SIZE)
        print('Episode:', episode, reward)
        eps *= eps_decay
