import numpy as np
import gym
import tensorflow as tf

tf.enable_eager_execution()

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
env = gym.wrappers.Monitor(env, "./videos/vid5", video_callable=lambda episode_id: True, force=True)

n_obs_params = env.observation_space.shape[0]
n_acts = env.action_space.n


def play_episode(net, render=False):
    observations = []
    actions = []
    rewards = []

    obs = env.reset()
    done = False

    while not done:
        if render: env.render()

        observations.append(obs)
        obs = np.expand_dims(obs, axis=0)
        logits = net(obs)
        act = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)[0].numpy()
        actions.append(act)
        next_obs, reward, done, info = env.step(act)
        rewards.append(reward)
        obs = next_obs

    return observations, actions, rewards


model = tf.keras.models.load_model('model/net.h5')
print(model)
play_episode(model, True)
