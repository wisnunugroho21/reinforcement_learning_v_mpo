# V-MPO
Simple code to demonstrate Deep Reinforcement Learning by using an on-policy adaptation of Maximum a Posteriori
Policy Optimization (MPO) in Pytorch

## Getting Started

This project is using Pytorch for Deep Learning Framework, Gym for Reinforcement Learning Environment.
Although it's not required, but i recommend run this project on a PC with GPU and 8 GB Ram

### Prerequisites

Make sure you have installed Pytorch and Gym.  
- Click [here](https://gym.openai.com/docs/) to install gym
- Click [here](https://pytorch.org/get-started/locally/) to install pytorch

### Installing

Just clone this project into your work folder

```
git clone https://github.com/wisnunugroho21/reinforcement_learning_v_mpo.git
```

## Running the project

After you clone the project, run following script in cmd/terminal :

#### Discrete
```
python discrete.py
```

#### Continous
```
python continous.py
```

## On-Policy adaptation of Maximum a Posteriori Policy Optimization (MPO)
Some of the most successful applications of deep reinforcement learning to chal-
lenging domains in discrete and continuous control have used policy gradient
methods in the on-policy setting. However, policy gradients can suffer from large
variance that may limit performance, and in practice require carefully tuned entropy
regularization to prevent policy collapse. As an alternative to policy gradient algo-
rithms, we introduce V-MPO, an on-policy adaptation of Maximum a Posteriori
Policy Optimization (MPO) that performs policy iteration based on a learned state-
value function. We show that V-MPO surpasses previously reported scores for both
the Atari-57 and DMLab-30 benchmark suites in the multi-task setting, and does so
reliably without importance weighting, entropy regularization, or population-based
tuning of hyperparameters. On individual DMLab and Atari levels, the proposed
algorithm can achieve scores that are substantially higher than has previously been
reported. V-MPO is also applicable to problems with high-dimensional, continuous
action spaces, which we demonstrate in the context of learning to control simulated
humanoids with 22 degrees of freedom from full state observations and 56 degrees
of freedom from pixel observations, as well as example OpenAI Gym tasks where
V-MPO achieves substantially higher asymptotic scores than previously reported.

You can read full detail of V-MPO in [here](https://arxiv.org/pdf/1909.12238.pdf)
