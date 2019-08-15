# reinforcement-learning

![Pong GIF](https://github.com/vaibhav369/reinforcement-learning/blob/master/results/pong_episode.gif)

## Description

This repository includes different reinforcement-learning projects pursued by myself while trying to learn the interesting field. The idea of training an agent solely by giving delayed outcomes is very fascinating. It lends itself very naturally to the real-world scenarios we come across. For example, stock trading given current prices, price lists and news information, can be treated as a very large reinforcement learning problem where the reward achieved is the profit/loss incurred. 

I describe my projects further in a brief manner


### bandit-problems

I start out with the "Hello - World" example of reinforcement learning problems. The problem statement is as follows :-
The agent is presented with n choices of levers, where every lever pull gives some reward. Now, every lever has a random reward attached with it, but the random value may oscillate about a mean governed by some ma thematical function, which is unknown to our agent. The goal of the problem is to fetch maximum possible reward. This problem helps understanding the exploration-exploitation dillema which is central to reinforcement learning.

### vanilla-policy-gradient

Reinforcement_learning to be of any practical use, has to be used with a function approximator, and deep-neural-networks are the best function approximators we have as of today. They are very general and apply to many different types of problems. Policy is a function within the agent which accepts environment observation as input and outputs actions. The algorithm computes the gradient of average reward gotten w.r.t agent's policy, and moves policy function in increasing direction of the gradient.

### q-learning

One of the standard techniques in deep-reinforcement learning is q-learning. q-values are the numerical values we attach to a particular state-action pair. So, whenever the agent finds itself in a state s, it looks at all possible actions, and takes the action with highest q value. Now, to learn q-values, we play many episodes and update our estimates using some methods, which are described better in code in the project.

The results of training the CartPole environment with a few different methods are shown in the images below :-

![Boltzmann Q Policy](https://github.com/vaibhav369/reinforcement-learning/blob/master/results/CartPole-v0_BoltzmannQPolicy.png)
![Epsilon Greedy Q Policy](https://github.com/vaibhav369/reinforcement-learning/blob/master/results/CartPole-v0_EpsGreedyQPolicy.png)
![Greedy Q Policy](https://github.com/vaibhav369/reinforcement-learning/blob/master/results/CartPole-v0_GreedyQPolicy.png)

Other results can be seen in the results tab.

### mountain-car

This is a very popular problem in the control theory, and has been adapted into a reinforcement learning environment by openai. The problem can be described as follows :- 
A car is stuck in a valley. It has to get out, but the engine does not have enough power to push through in one go. So the agent has to devise a strategy to get out, by optimally using its engine to go forward, backward and (not push at all), according to what situation the car is in.

![Mountain Car GIF](https://github.com/vaibhav369/reinforcement-learning/blob/master/mountain-car/results/episode.gif)

### pong

pong is again an openai environment. Our agent is a paddle which can move upwards and downwards on the screen. The goal is to obviously not let the ball pass through on agent's side, and make the ball pass through the opponent's paddle. For solving the pong environment, we use a DQN again. the GIF after training is displayed at the top of readme :)
