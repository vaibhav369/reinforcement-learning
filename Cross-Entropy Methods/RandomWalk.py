import matplotlib.pyplot as plt
import random

class Env:

    def __init__(self):
        self.num_states = 7
        self.num_actions = 2
        self.reset()

    def step(self, action):
        assert action == 1 or action == -1
        self.state += action
        reward = 0
        episode_done = False

        if self.state == -4:
            reward = -1
            episode_done = True
        elif self.state == 4:
            reward = 1
            episode_done = True

        self.timestep += 1
        if self.timestep >= 100:
            episode_done = True

        return (self.state, reward, episode_done)

    def reset(self):
        self.state = 0
        self.timestep = 0
        return self.state


def get_policy1(num_states, num_actions):
    # this is a random policy
    return random.choice( [(-1, 1) for state in range(-3, 4)] )

def get_action(state):
    return random.choice([-1, 1])

class Agent:

    def __init__(self, env):
        self.env = env
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        #self.policy = get_policy1(num_states, num_actions)
        self.values = {state: 0 for state in range(-3, 4)}
        self.num_visits = {state: 0 for state in range(-3, 4)}

    def play_episode(self):
        trajectory = []
        state = self.env.reset()

        episode_reward = 0
        episode_done = False
        while not episode_done:

            action = get_action(state)
            #action = self.policy[state]
            next_state, reward, episode_done = self.env.step(action)
            trajectory.append( (state, action, reward, next_state) )
            episode_reward += reward
            state = next_state
            #print(trajectory[-1])

        return trajectory

    def update_values(self):
        trajectory = self.play_episode()
        value = 0
        gamma = 0.95

        for state, action, reward, next_state in trajectory[::-1]:
            value = gamma * value + reward
            self.num_visits[state] += 1
            self.values[state] = self.values[state] + (value - self.values[state]) / self.num_visits[state]


env = Env()
agent = Agent(env)

NUM_EPISODES = 1000000

for episode in range(NUM_EPISODES):

    if episode % 100000 == 0:
        plt.plot(range(-3, 4), [agent.values[state] for state in range(-3, 4)])
        plt.title('Episode ' + str(episode))
        plt.show()

    agent.update_values()
