from collections import deque, namedtuple
import numpy as np


experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience_record):
        self.buffer.append(experience_record)

    def sample(self, batch_size):
        indices = np.random.choice( len(self.buffer), batch_size, replace=False )
        return [self.buffer[idx] for idx in indices]



if __name__ == '__main__':

    import gym

    experience_buffer = ExperienceBuffer(5)

    env = gym.make('CartPole-v0')
    state = env.reset()
    done = False


    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        exp = experience(state, action, reward, done, next_state)
        experience_buffer.append(exp)
        print(experience_buffer.buffer)

        if len(experience_buffer) > 2:
            print(experience_buffer.sample(2))
