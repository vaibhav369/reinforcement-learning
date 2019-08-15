import random
import torch


def get_action(state, env, function_approximator, eps):
    if random.random() < eps:
        return env.action_space.sample()
    state_vector = to_torch_float_tensor([state])
    q_vals = function_approximator(state_vector)
    max_q, best_action = torch.max(q_vals, dim=1)
    return int(best_action.item())


def to_torch_float_tensor(lst):
    return torch.tensor(lst, dtype=torch.float32)


if __name__ == '__main__':

    import gym
    from function_approximators import DQN

    env = gym.make('CartPole-v0')
    function_approximator = DQN(env.observation_space.shape[0], env.action_space.n)
    eps = 1

    state = env.reset()
    done = False

    while not done:
        action = get_action(state, env, function_approximator, eps)
        next_state, reward, done, info = env.step(action)
        state = next_state
