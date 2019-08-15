import torch
import torch.nn as nn



class DQN(nn.Module):

    def __init__(self, n_state_features, n_actions):
        super(DQN, self).__init__()
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Linear(n_state_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, state):
        assert self.n_state_features == state.shape[1]      # 0th dimension is for batch
        return self.network(state)



if __name__ == '__main__':

    import numpy as np

    n_state_features = 4
    n_actions = 2

    function_approximator = DQN(n_state_features, n_actions)

    a = np.array([1, 2, 3, 4], dtype=np.float32)
    v = torch.tensor(a)

    print( function_approximator(v) )
