import torch
from torch import nn
from torch import optim
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter


writer = SummaryWriter()

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(obs_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def get_action(observation, net):
    observation_v = torch.FloatTensor([observation])
    sm = nn.Softmax(dim=1)
    action_prob_v = sm( net(observation_v) )
    action_prob = action_prob_v.data.numpy()[0]
    action = np.random.choice(len(action_prob), p=action_prob)
    return action


def run_episode(env, net, render=False):
    observation = env.reset()
    done = False
    episode_reward = 0.0
    episode_steps = []

    while not done:
        action = get_action(observation, net)
        next_observation, reward, done, info = env.step(action)
        episode_reward += reward
        episode_steps.append( EpisodeStep(observation=observation, action=action) )
        observation = next_observation
        if render: env.render()

    return episode_reward, episode_steps


def iterate_batches(env, net, batch_size, num_episodes):
    batch = []
    for episode in range(num_episodes):
        episode_reward, episode_steps = run_episode(env, net)
        batch.append(Episode(reward=episode_reward, steps=episode_steps))
        if len(batch) == batch_size:
            yield batch
            batch = []
        print('Episode:', episode, 'Reward:', episode_reward)


def filter_batch(batch, percentile):
    batch.sort(key=lambda episode: episode.reward)
    batch.reverse()
    return batch[:int(len(batch) * percentile/100)]


def train(env, net, objective, optimizer):
    NUM_EPISODES = 1000
    BATCH_SIZE = 16
    PERCENTILE = 50

    for batch_number, batch in enumerate(iterate_batches(env, net, BATCH_SIZE, NUM_EPISODES)):
        mean_reward = sum([episode.reward for episode in batch]) / len(batch)
        #writer.add_scalar('rewards_per_batch', mean_reward, batch_number)

        filtered_batch = filter_batch(batch, PERCENTILE)

        train_observation = []
        train_action = []

        for episode in filtered_batch:
            train_observation.extend([step.observation for step in episode.steps])
            train_action.extend([step.action for step in episode.steps])

        _train(net, train_observation, train_action, objective, optimizer, batch_number)



def _train(net, train_observation, train_action, objective, optimizer, batch_number):
    train_observation_v = torch.FloatTensor(train_observation)
    train_action_v = torch.LongTensor(train_action)
    optimizer.zero_grad()
    action_v = net(train_observation_v)
    loss_v = objective(action_v, train_action_v)
    loss_v.backward()
    optimizer.step()
    #writer.add_scalar('loss_per_batch', loss_v.item(), batch_number)


if __name__ == '__main__':

    HIDDEN_SIZE = 128

    env = gym.make('CartPole-v0')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    #writer = SummaryWriter()

    train(env, net, objective, optimizer)

    run_episode(env, net, render=True)
    writer.close()
