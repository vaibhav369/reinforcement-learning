# reinforcement-learning

## Description

This repository includes different reinforcement-learning projects pursued by myself while trying to learn the interesting field. The idea of training an agent solely by giving delayed outcomes is very fascinating. It lends itself very naturally to the real-world scenarios we come across. For example, stock trading given current prices, price lists and news information, can be treated as a very large reinforcement learning problem where the reward achieved is the profit/loss incurred. 

I describe my projects further in a brief manner


### bandit-problems

I start out with the "Hello - World" example of reinforcement learning problems. The problem statement is as follows :-
The agent is presented with n choices of levers, where every lever pull gives some reward. Now, every lever has a random reward attached with it, but the random value may oscillate about a mean governed by some ma thematical function, which is unknown to our agent. The goal of the problem is to fetch maximum possible reward. This problem helps understanding the exploration-exploitation dillema which is central to reinforcement learning.

### vanilla-policy-gradient

Reinforcement_learning to be of any practical use, has to be used with a function approximator, and deep-neural-networks are the best function approximators we have as of today. They are very general and apply to many different types of problems. Policy is a function within the agent which accepts environment observation as input and outputs actions. The algorithm computes the gradient of average reward gotten w.r.t agent's policy, and moves policy function in increasing direction of the gradient.


