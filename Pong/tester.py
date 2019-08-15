import gym
import time
import argparse
import numpy as np
import torch
import wrappers
import dqn_model


ENV_NAME = 'PongNoFrameskip-v4'
FPS = 25


if __name__ == '__main__':

    env = wrappers.make_env(ENV_NAME)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load('models/PongNoFrameskip-v4-best.dat', map_location='cpu'))

    total_reward = 0.0
    state = env.reset()
    done = False

    while not done:
        start_ts = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        next_state, reward, done, info = env.step(action)
        state = next_state

        total_reward += reward

        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)

    print( 'Total reward: %.2f' %total_reward )
